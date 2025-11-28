import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
import torch.nn as nn

# Add current directory to path
sys.path.append(os.path.abspath("."))

import ellice

def load_and_preprocess_data():
    data_raw = load_breast_cancer()
    X = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
    y = pd.Series(data_raw.target, name="target")

    # Scale Data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, data_raw.feature_names

def print_model_metrics(y_true, y_pred, y_prob, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob)
    print(f"\n{name} Metrics on Test Set:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Log Loss: {loss:.4f}")

def print_result(cf, query, data, target):
    if cf.empty:
        print("Failed to generate CF.")
        return

    cf_row = cf.iloc[0]
    
    # Calculate stats
    l2_dist = np.linalg.norm(cf_row[data.feature_names] - query[data.feature_names])
    
    print(f"CF Found! L2 Dist: {l2_dist:.4f}")
    print(f"Original X (first 5): {query.head().values}")
    print(f"CF X (first 5):       {cf_row[data.feature_names].head().values}")
    
    print("\nProbabilities:")
    
    model_prob_target = cf_row['model_prob_class_1'] if target == 1 else 1 - cf_row['model_prob_class_1']
    print(f"  Model Prob (Target {target}):      {model_prob_target:.4f}")
    print(f"  Worst Case Prob (Target {target}): {cf_row['worst_case_prob_target']:.4f}")
    
    if cf_row['worst_case_prob_target'] > 0.5:
        print("  -> Robustly flipped!")
    else:
        print("  -> Not robustly flipped (prob <= 0.5)")

def test_sklearn_integration():
    print("\n" + "="*50)
    print("Testing Sklearn Integration (Breast Cancer Dataset)")
    print("="*50)
    
    X_scaled, y, feature_names = load_and_preprocess_data()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Sklearn Model
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print_model_metrics(y_test, y_pred, y_prob, name="Sklearn LR")
    
    # Setup ElliCE
    full_df = X_train.copy()
    full_df['target'] = y_train
    
    data = ellice.Data(full_df, target_column='target')
    exp = ellice.Explainer(clf, data, backend='sklearn')
    
    # Select a query instance
    query = X_test.iloc[0]
    orig_pred_prob = clf.predict_proba([query])[0]
    orig_pred = int(orig_pred_prob[1] > 0.5)
    target = 1 - orig_pred
    
    print(f"\nOriginal Instance Index: {query.name}")
    print(f"Original Prediction: {orig_pred} (Prob Class 1: {orig_pred_prob[1]:.4f})")
    print(f"Target Class: {target}")

    # Test Range of Epsilons
    epsilons = [0.01] # [0.01, 0.02, 0.03, 0.04, 0.05]
    
    for eps in epsilons:
        print(f"\n" + "-"*30)
        print(f"Generating CF with Epsilon: {eps}")
        print("-"*30)
        
        cf = exp.generate_counterfactuals(
            query, 
            method='continuous', 
            target_class=target,
            robustness_epsilon=eps,
            regularization_coefficient=0.01,
            return_probs=True,
            progress_bar=True,
            optimization_params={'learning_rate': 0.001, 'max_iterations': 2000, 'robustness_weight': 10.0, 'early_stopping': True, 'patience': 200}
        )
        
        print_result(cf, query, data, target)

def test_pytorch_integration():
    print("\n" + "="*50)
    print("Testing PyTorch Integration (Breast Cancer Dataset)")
    print("="*50)
    
    X_scaled, y, feature_names = load_and_preprocess_data()
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to Torch
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    # Train Simple Torch Model
    input_dim = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training PyTorch Model...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_prob = torch.sigmoid(logits).numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)
        
    print_model_metrics(y_test, y_pred, y_prob, name="PyTorch MLP")
        
    # Setup ElliCE
    full_df = X_train.copy()
    full_df['target'] = y_train
    
    data = ellice.Data(full_df, target_column='target')
    exp = ellice.Explainer(model, data, backend='pytorch')
    
    # Select a query instance
    query = X_test.iloc[0]
    # Predict
    with torch.no_grad():
        logit = model(torch.tensor(query.values, dtype=torch.float32).unsqueeze(0))
        prob = torch.sigmoid(logit).item()
    
    orig_pred = int(prob > 0.5)
    target = 1 - orig_pred
    
    print(f"\nOriginal Prediction: {orig_pred} (Prob Class 1: {prob:.4f})")
    print(f"Target Class: {target}")
    
    print("Generating Continuous CF (Torch)...")
    cf = exp.generate_counterfactuals(
        query,
        method='continuous',
        target_class=target, 
        return_probs=True,
        progress_bar=True,
        robustness_epsilon=0.01,
        regularization_coefficient=0.0001,
        optimization_params={
            'learning_rate': 0.001, 
            'max_iterations': 2000, 
            'early_stopping': True,
            'gradient_mode': 'min-max'
        }
    )
    print_result(cf, query, data, target)

if __name__ == "__main__":
    try:
        test_sklearn_integration()
        test_pytorch_integration()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed with error: {e}")
        import traceback
        traceback.print_exc()
