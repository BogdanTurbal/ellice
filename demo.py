import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ellice

# ---------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------------------------------------------------------
# 1. Data Loading & Preparation
# ---------------------------------------------------------
print("--- [1] Loading Breast Cancer Dataset ---")
data_raw = load_breast_cancer()
X = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
y = pd.Series(data_raw.target, name="target")

# Standardize features
scaler = StandardScaler()
X_scaled = X#pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define constraints
# 'worst concavity' is immutable (cancer characteristic that doesn't change easily)
# 'mean radius' cannot decrease (tumors don't shrink without treatment, we assume actionable change shouldn't rely on this)
# 'mean texture' restricted range (assuming measurement validity)
one_way_change = {'mean radius': 'increase'}
features_to_vary = [col for col in X.columns if col != 'worst concavity']
permitted_range = {'mean texture': [10.0, 30.0]} # Restrict to plausible texture values (e.g. 10-30)

print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")

# ---------------------------------------------------------
# 2. Model Training (PyTorch MLP)
# ---------------------------------------------------------
print("\n--- [2] Training Neural Network ---")
input_dim = X_train.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 1) # Binary classification
)

# Training loop
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
X_train_t = torch.FloatTensor(X_train.values)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()

# Evaluate
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test.values)
    preds = (torch.sigmoid(model(X_test_t)) > 0.5).numpy().flatten()
    acc = (preds == y_test.values).mean()
    print(f"Model Test Accuracy: {acc:.4f}")

# ---------------------------------------------------------
# 3. ElliCE Setup & Generation
# ---------------------------------------------------------
print("\n--- [3] Initializing ElliCE ---")
full_df = X_train.copy()
full_df['target'] = y_train
data = ellice.Data(full_df, target_column='target')

exp = ellice.Explainer(
    model=model,
    data=data,
    backend='pytorch'
)

# Select a query instance (e.g., predicted as 0/Malignant, want to flip to 1/Benign)
# In this dataset 0 = Malignant, 1 = Benign
idx = 0
query = X_test.iloc[idx]
with torch.no_grad():
    logit = model(torch.FloatTensor(query.values).unsqueeze(0))
    prob = torch.sigmoid(logit).item()
    pred_class = int(prob > 0.5)

target_class = 1 - pred_class

print(f"\nQuery Instance (Index {idx}):")
print(f"Prediction: {pred_class} (Prob: {prob:.4f})")
print(f"Target Class: {target_class}")

print("\n--- [4] Generating Robust Actionable Counterfactual ---")
cf = exp.generate_counterfactuals(
    query,
    target_class=target_class,
    method='continuous',
    features_to_vary=features_to_vary,
    permitted_range=permitted_range,
    one_way_change=one_way_change,
    robustness_epsilon=0.01,
    return_probs=True,
    progress_bar=True,
    optimization_params={'max_iterations': 1000}
)

if not cf.empty:
    print("\n--- Result Found ---")
    res = cf.iloc[0]
    print(f"Robust Probability (Target {target_class}): {res['worst_case_prob_target']:.4f}")
    
    # Verify Constraints
    # 1. Immutable
    orig_imm = query['worst concavity']
    new_imm = res['worst concavity']
    print(f"Immutable 'worst concavity': Orig={orig_imm:.4f}, CF={new_imm:.4f} (Diff={abs(new_imm-orig_imm):.4f})")
    
    # 2. Range
    val_range = res['mean texture']
    allowed_range = permitted_range.get('mean texture', 'Any')
    print(f"Range 'mean texture': {val_range:.4f} (Allowed: {allowed_range})")
    
    # 3. One-way
    orig_ow = query['mean radius']
    new_ow = res['mean radius']
    print(f"One-way 'mean radius': Orig={orig_ow:.4f}, CF={new_ow:.4f} (Should >= Orig)")
    
    print("\nFeature Changes:")
    # Print top 5 changes
    diffs = (cf[data.feature_names].iloc[0] - query).abs().sort_values(ascending=False)
    for feat in diffs.index:
        print(f"  {feat}: {query[feat]:.4f} -> {cf.iloc[0][feat]:.4f}")
else:
    print("\nFailed to generate counterfactual.")

