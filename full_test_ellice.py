import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

# Add the directory containing the 'ellice' package to sys.path
# Assuming this script is in paper_project/ellice/
# We need paper_project/ellice/ to be in sys.path so that 'import ellice' finds the inner ellice package
current_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in paper_project/ellice/, current_dir is paper_project/ellice/
# We want to import 'ellice' which is inside this directory.
# So we add current_dir to sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import ellice

def create_synthetic_data(n_samples=1000, n_features=10):
    # 1. Create base numerical data
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 2. Convert to DataFrame
    feature_names = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # 3. Create a categorical feature (from feat_0)
    # Bin feat_0 into 3 categories: Low, Medium, High
    df['cat_0'] = pd.cut(df['feat_0'], bins=3, labels=["Low", "Medium", "High"])
    
    # Drop original feat_0
    df = df.drop(columns=['feat_0'])
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['cat_0'], prefix='cat', dtype=float)
    
    # Identify one-hot columns
    cat_cols = [col for col in df_encoded.columns if col.startswith('cat_')]
    
    # Identify continuous columns
    cont_cols = [col for col in df_encoded.columns if col.startswith('feat_')]
    
    return df_encoded, pd.Series(y, name="target"), cat_cols, cont_cols

def print_result(cf, query, data, target, description=""):
    print(f"\n--- Result: {description} ---")
    if cf.empty:
        print("Failed to generate CF.")
        return

    cf_row = cf.iloc[0]
    
    # Check robustness
    robust_prob = cf_row['worst_case_prob_target']
    print(f"Worst Case Prob (Target {target}): {robust_prob:.4f}")
    if robust_prob > 0.5:
        print("  -> Robustly flipped! [PASS]")
    else:
        print("  -> Not robustly flipped (prob <= 0.5) [FAIL]")
        
    # Check actionability
    # 1. One-hot constraints
    # Check if cat features sum to 1
    # Assuming we know the group
    
    # 2. Feature changes
    diff = cf_row[data.feature_names] - query[data.feature_names]
    changed_features = diff[abs(diff) > 1e-4].index.tolist()
    print(f"Changed features: {changed_features}")
    
    return cf_row

def test_all_functionality():
    print("\n" + "="*60)
    print("ELLICE: Full Functionality Test")
    print("="*60)
    
    # 1. Data Setup
    print("\n[1] Creating Synthetic Data with Categorical Features...")
    X, y, cat_cols, cont_cols = create_synthetic_data()
    one_hot_groups = [cat_cols] # List of lists
    print(f"Features: {X.columns.tolist()}")
    print(f"Categorical Groups: {one_hot_groups}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Model Training
    print("\n[2] Training Logistic Regression...")
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    print(f"Test Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
    
    # 3. Setup ElliCE
    full_df = X_train.copy()
    full_df['target'] = y_train
    data = ellice.Data(full_df, target_column='target')
    exp = ellice.Explainer(clf, data, backend='sklearn')
    
    # Select query
    query = X_test.iloc[0]
    orig_pred = clf.predict([query])[0]
    target = 1 - orig_pred
    print(f"\nQuery Index: {query.name}, Original Pred: {orig_pred}, Target: {target}")
    print(f"Query Categorical Values: {query[cat_cols].values}")
    
    # --- Test 1: Continuous Generator (Standard) ---
    print("\n[3.1] Continuous Generator: Standard")
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_hot_groups=one_hot_groups,
        return_probs=True,
        progress_bar=False,
        optimization_params={'max_iterations': 500, 'learning_rate': 0.05}
    )
    res = print_result(cf, query, data, target, "Standard Continuous")
    
    # Check One-Hot Integrity
    if not cf.empty:
        cat_sum = cf.iloc[0][cat_cols].sum()
        print(f"Sum of one-hot group: {cat_sum:.4f} (Should be 1.0)")
        if not np.isclose(cat_sum, 1.0):
             print("  -> One-Hot constraint VIOLATED! [FAIL]")
        else:
             print("  -> One-Hot constraint SATISFIED! [PASS]")

    # --- Test 2: Actionability - Immutability ---
    print("\n[3.2] Continuous Generator: Immutability")
    # Freeze all continuous features, only allow categorical changes
    # OR freeze the most important continuous feature
    feat_to_freeze = cont_cols[0]
    features_to_vary = [c for c in X.columns if c != feat_to_freeze]
    
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        features_to_vary=features_to_vary,
        one_hot_groups=one_hot_groups,
        return_probs=True,
        progress_bar=False,
        optimization_params={'max_iterations': 500}
    )
    res = print_result(cf, query, data, target, f"Immutable {feat_to_freeze}")
    if not cf.empty:
        if abs(res[feat_to_freeze] - query[feat_to_freeze]) < 1e-4:
            print(f"  -> Feature {feat_to_freeze} remained unchanged. [PASS]")
        else:
            print(f"  -> Feature {feat_to_freeze} CHANGED! [FAIL]")

    # --- Test 3: Actionability - Ranges ---
    print("\n[3.3] Continuous Generator: Ranges")
    # Restrict feat_1 to be within [0, 0.5] (assuming standardized or rough scale)
    # Let's check query value first
    q_val = query[cont_cols[1]]
    print(f"Query {cont_cols[1]}: {q_val:.4f}")
    
    permitted_range = {cont_cols[1]: [-10.0, q_val + 0.1]} # Can't increase much
    
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        permitted_range=permitted_range,
        one_hot_groups=one_hot_groups,
        return_probs=True,
        progress_bar=False,
        optimization_params={'max_iterations': 500}
    )
    res = print_result(cf, query, data, target, "Range Constraint")
    if not cf.empty:
        val = res[cont_cols[1]]
        min_v, max_v = permitted_range[cont_cols[1]]
        # Use small tolerance
        if min_v - 1e-5 <= val <= max_v + 1e-5:
             print(f"  -> Feature {cont_cols[1]} value {val:.4f} in range. [PASS]")
        else:
             print(f"  -> Feature {cont_cols[1]} value {val:.4f} OUT of range! [FAIL]")

    # --- Test 4: Actionability - One-Way ---
    print("\n[3.4] Continuous Generator: One-Way Change")
    # Force feat_2 to only increase
    one_way = {cont_cols[2]: 'increase'}
    q_val_2 = query[cont_cols[2]]
    print(f"Query {cont_cols[2]}: {q_val_2:.4f}")
    
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='continuous',
        one_way_change=one_way,
        one_hot_groups=one_hot_groups,
        return_probs=True,
        progress_bar=False,
        optimization_params={'max_iterations': 500}
    )
    res = print_result(cf, query, data, target, "One-Way Increase")
    if not cf.empty:
        val = res[cont_cols[2]]
        if val >= q_val_2 - 1e-5:
             print(f"  -> Feature {cont_cols[2]} increased or stayed same ({val:.4f}). [PASS]")
        else:
             print(f"  -> Feature {cont_cols[2]} decreased ({val:.4f})! [FAIL]")

    # --- Test 5: Discrete Generator (Data Supported) ---
    print("\n[3.5] Discrete Generator: Standard")
    # Discrete generator should just pick a point from data
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='discrete',
        one_hot_groups=one_hot_groups, # Should be ignored but harmless
        return_probs=True,
        progress_bar=False
    )
    res = print_result(cf, query, data, target, "Discrete Generator")
    
    # --- Test 6: Discrete Generator Constraints ---
    print("\n[3.6] Discrete Generator: Constraints")
    # Freeze a categorical feature which is more likely to have matches
    # cat_cols[0] is likely 'cat_Low' (0 or 1)
    feat_to_freeze = cat_cols[0]
    features_to_vary = [c for c in X.columns if c != feat_to_freeze]
    
    print(f"Freezing {feat_to_freeze} (Value: {query[feat_to_freeze]})")
    
    cf = exp.generate_counterfactuals(
        query, target_class=target,
        method='discrete',
        features_to_vary=features_to_vary,
        return_probs=True,
        progress_bar=False
    )
    res = print_result(cf, query, data, target, f"Discrete Immutable {feat_to_freeze}")
    if not cf.empty:
        if abs(res[feat_to_freeze] - query[feat_to_freeze]) < 1e-4:
            print(f"  -> Feature {feat_to_freeze} matched original. [PASS]")
        else:
            print(f"  -> Feature {feat_to_freeze} mismatch! [FAIL]")

if __name__ == "__main__":
    try:
        test_all_functionality()
        print("\nAll Full Tests Completed!")
    except Exception as e:
        print(f"\nTests failed with error: {e}")
        import traceback
        traceback.print_exc()

