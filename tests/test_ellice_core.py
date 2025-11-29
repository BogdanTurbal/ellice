import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os
import sys

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add package root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ellice

# Load Config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.fixture(scope="module")
def setup_data_model():
    seed_everything(CONFIG['random_state'])
    # Create synthetic data based on config
    X, y = make_classification(
        n_samples=CONFIG['samples'], 
        n_features=CONFIG['features'], 
        n_informative=CONFIG['informative'], 
        n_redundant=CONFIG['redundant'], 
        class_sep=CONFIG['class_sep'],
        random_state=CONFIG['random_state']
    )
    feature_names = [f"feat_{i}" for i in range(CONFIG['features'])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Create categorical feature (feat_9 split into 3 bins)
    df['cat_0'] = pd.cut(df[f"feat_{CONFIG['features']-1}"], bins=3, labels=["Low", "Medium", "High"])
    df = df.drop(columns=[f"feat_{CONFIG['features']-1}"])
    df_encoded = pd.get_dummies(df, columns=['cat_0'], prefix='cat', dtype=float)
    
    cat_cols = [col for col in df_encoded.columns if col.startswith('cat_')]
    cont_cols = [col for col in df_encoded.columns if col.startswith('feat_')]
    
    # Add an ordinal feature (0-9)
    # We can just reuse a random column or add a new one
    # Let's replace feat_8 with an ordinal version
    df_encoded['ordinal_feat'] = np.random.randint(0, 10, size=len(df_encoded))
    
    X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])
    
    # Use Logistic Regression (Sklearn)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Compute metrics
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Report metrics via logger (will be saved to file)
    logger.info(f"Sklearn Model - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Use PyTorch Model
    input_dim = X_train.shape[1]
    torch_model = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Proper training loop
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    torch_model.train()
    for _ in range(50): # Few epochs to get decent weights
        optimizer.zero_grad()
        outputs = torch_model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Quick eval for torch model (untrained, just random)
    torch_model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test.values)
        logits = torch_model(X_t)
        preds = (torch.sigmoid(logits) > 0.5).numpy().flatten().astype(int)
        torch_acc = (preds == y_test).mean()
        logger.info(f"PyTorch Model (Random) - Test Acc: {torch_acc:.4f}")

    # Setup ElliCE Data
    full_df = X_train.copy()
    full_df['target'] = y_train
    data = ellice.Data(full_df, target_column='target')
    
    return {
        'sklearn_model': model,
        'torch_model': torch_model,
        'data': data,
        'X_test': X_test,
        'cat_cols': cat_cols,
        'cont_cols': cont_cols
    }

@pytest.fixture(params=['sklearn', 'pytorch'])
def explainer(request, setup_data_model):
    backend = request.param
    if backend == 'sklearn':
        return ellice.Explainer(
            model=setup_data_model['sklearn_model'],
            data=setup_data_model['data'],
            backend='sklearn'
        )
    else:
        return ellice.Explainer(
            model=setup_data_model['torch_model'],
            data=setup_data_model['data'],
            backend='pytorch'
        )

@pytest.fixture
def query_instance(explainer, setup_data_model):
    X_test = setup_data_model['X_test']
    
    # Find a suitable query point
    # For sklearn, we can predict. For torch, we wrap it or just pick index 0
    # To be consistent across parameterized tests, we pick one instance and check its prediction
    # using the CURRENT explainer's model wrapper
    
    idx = 0
    query = X_test.iloc[idx]
    
    # Get prediction from explainer's model wrapper
    # Note: For PyTorch models, this might move the model to GPU if available.
    # But query.to_frame().T.values is numpy.
    probs = explainer.model.predict_proba(query.to_frame().T.values)[0]
    # Class 1 prob is at index 1
    pred_class = 1 if probs[1] > 0.5 else 0
    target = 1 - pred_class
    
    return query, target

def test_basic_continuous_generation(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_hot_groups=[cat_cols],
        return_probs=True,
        progress_bar=False,
        optimization_params={'max_iterations': CONFIG['max_iterations']}
    )
    
    assert not cf.empty, "Counterfactual should be generated"
    
    # Check validity
    model_prob = cf['model_prob_class_1'].iloc[0]
    pred_class = 1 if model_prob > 0.5 else 0
    assert pred_class == target, f"CF should flip prediction to {target}"
    
    # Check One-Hot Integrity: EXACTLY ONE value should be 1.0, others 0.0
    # (within float tolerance)
    cat_vals = cf[cat_cols].iloc[0].values
    # Check sum
    assert np.isclose(cat_vals.sum(), 1.0, atol=CONFIG['float_tolerance']), "One-hot group must sum to 1"
    # Check individual values are close to 0 or 1
    is_binary = np.all([np.isclose(v, 0.0, atol=CONFIG['float_tolerance']) or np.isclose(v, 1.0, atol=CONFIG['float_tolerance']) for v in cat_vals])
    assert is_binary, f"Categorical values must be 0 or 1: {cat_vals}"

def test_robustness_guarantee(explainer, query_instance, setup_data_model):
    """Test that Robust Logit <= Model Logit (for target class 1)"""
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    
    # Use provided robustness epsilon
    eps = CONFIG['robustness_epsilon']
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_hot_groups=[cat_cols],
        return_probs=True,
        progress_bar=False,
        robustness_epsilon=eps
    )
    
    if cf.empty:
        pytest.skip("Could not find counterfactual")
    
    p_model = cf['model_prob_class_1'].iloc[0]
    p_robust = cf['worst_case_prob_target'].iloc[0]
    
    # Convert p_robust (prob of TARGET class) to logit for target class
    logit_robust = np.log(p_robust / (1 - p_robust))
    
    # Model logit for target class
    p_model_target = p_model if target == 1 else (1 - p_model)
    logit_model = np.log(p_model_target / (1 - p_model_target))
    
    # Robust logit (worst case) should be <= Model logit (best estimate)
    assert logit_robust <= logit_model + CONFIG['float_tolerance'], "Robust logit should be <= Model logit"

def test_small_epsilon_robustness(explainer, query_instance, setup_data_model):
    """
    Test with target eps 0.001.
    Check that robust logit corresponds to target class (is positive).
    This verifies we actually found a robust solution.
    """
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    
    eps = CONFIG['small_epsilon']
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_hot_groups=[cat_cols],
        return_probs=True,
        progress_bar=False,
        robustness_epsilon=eps,
        optimization_params={'max_iterations': CONFIG['max_iterations']}
    )
    
    if cf.empty:
        # If optimization failed completely, fail
        assert False, "Optimization failed to produce any result"
        
    p_robust = cf['worst_case_prob_target'].iloc[0]
    logit_robust = np.log(p_robust / (1 - p_robust))
    
    # Check if robust logit > 0 (i.e. robustly classified as target)
    # Note: With very small epsilon, model and robust are close. 
    # If we found a valid CF, logit_model > 0. 
    # logit_robust might be slightly < 0 if we are right on boundary, 
    # but ideally > 0 if optimization succeeded.
    
    # We check if prediction is robustly flipped (prob > 0.5)
    assert p_robust > 0.5, f"With small epsilon {eps}, CF should be robustly flipped (Prob: {p_robust:.4f})"

def test_immutability_continuous(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    feat_to_freeze = 'feat_0'
    
    features_to_vary = [c for c in query.index if c != feat_to_freeze]
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        features_to_vary=features_to_vary,
        one_hot_groups=[cat_cols],
        progress_bar=False
    )
    
    if not cf.empty:
        assert np.isclose(cf[feat_to_freeze].iloc[0], query[feat_to_freeze], atol=CONFIG['float_tolerance']), \
            f"Feature {feat_to_freeze} should remain unchanged"

def test_range_constraints_continuous(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    feat = 'feat_1'
    
    permitted_range = {feat: [-0.5, 0.5]}
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        permitted_range=permitted_range,
        one_hot_groups=[cat_cols],
        progress_bar=False
    )
    
    if not cf.empty:
        val = cf[feat].iloc[0]
        assert -0.5 - CONFIG['float_tolerance'] <= val <= 0.5 + CONFIG['float_tolerance'], \
            f"Feature {feat} value {val} outside range [-0.5, 0.5]"

def test_one_way_constraint_continuous(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    feat = 'feat_2'
    orig_val = query[feat]
    
    one_way = {feat: 'increase'}
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_way_change=one_way,
        one_hot_groups=[cat_cols],
        progress_bar=False
    )
    
    if not cf.empty:
        val = cf[feat].iloc[0]
        assert val >= orig_val - CONFIG['float_tolerance'], \
            f"Feature {feat} should increase or stay same (Orig: {orig_val}, CF: {val})"

def test_allowed_values_continuous(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    feat = 'feat_3'
    
    allowed = {feat: [-1.0, 0.0, 1.0]}
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        allowed_values=allowed,
        one_hot_groups=[cat_cols],
        progress_bar=False
    )
    
    if not cf.empty:
        val = cf[feat].iloc[0]
        is_allowed = any(np.isclose(val, av, atol=CONFIG['float_tolerance']) for av in allowed[feat])
        assert is_allowed, f"Feature {feat} value {val} is not in allowed set {allowed[feat]}"

def test_discrete_generator_basic(explainer, query_instance):
    query, target = query_instance
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='data_supported',
        progress_bar=False
    )
    
    if cf.empty:
        pytest.skip("No discrete candidate found")

def test_discrete_generator_constraints(explainer, query_instance):
    query, target = query_instance
    feat = 'ordinal_feat'
    
    features_to_vary = [c for c in query.index if c != feat]
    
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='data_supported',
        features_to_vary=features_to_vary,
        progress_bar=False
    )
    
    if not cf.empty:
        assert np.isclose(cf[feat].iloc[0], query[feat], atol=CONFIG['float_tolerance']), \
            f"Discrete: Feature {feat} should match original"
    else:
        pytest.skip("No discrete candidate found satisfying constraints")

def test_sparse_continuous_generation(explainer, query_instance, setup_data_model):
    query, target = query_instance
    cat_cols = setup_data_model['cat_cols']
    
    # Force sparsity
    cf = explainer.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_hot_groups=[cat_cols],
        return_probs=True,
        progress_bar=False,
        sparsity=True,
        optimization_params={'max_iterations': CONFIG['max_iterations']}
    )
    
    if cf.empty:
        pytest.skip("Optimization did not find robust sparse CF in limited steps")
        
    # Check basic validity
    model_prob = cf['model_prob_class_1'].iloc[0]
    pred_class = 1 if model_prob > 0.5 else 0
    assert pred_class == target

def test_data_supported_search_modes(explainer, query_instance):
    query, target = query_instance
    
    # Test Filtering
    cf_filt = explainer.generate_counterfactuals(
        query, target_class=target,
        method='data_supported',
        search_mode='filtering',
        progress_bar=False
    )
    
    # Test KDTree (only if sparse=False)
    cf_kdtree = explainer.generate_counterfactuals(
        query, target_class=target,
        method='data_supported',
        search_mode='kdtree',
        sparsity=False,
        progress_bar=False
    )
    
    # Test BallTree (only if sparse=True)
    cf_ball = explainer.generate_counterfactuals(
        query, target_class=target,
        method='data_supported',
        search_mode='ball_tree',
        sparsity=True,
        progress_bar=False
    )
    
    # Just ensure no errors raised and types are correct
    # Results might depend on data availability
    
def test_data_supported_invalid_modes(explainer, query_instance):
    query, target = query_instance
    
    # KDTree with sparsity=True should fail
    # This fails immediately due to parameter validation
    with pytest.raises(ValueError):
        explainer.generate_counterfactuals(
            query, target_class=target,
            method='data_supported',
            search_mode='kdtree',
            sparsity=True,
            progress_bar=False
        )

    # BallTree with sparsity=False should fail
    with pytest.raises(ValueError):
        explainer.generate_counterfactuals(
            query, target_class=target,
            method='data_supported',
            search_mode='ball_tree',
            sparsity=False,
            progress_bar=False
        )

