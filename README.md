# ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets

ElliCE provides robust counterfactual explanations that remain valid across the set of all nearly-optimal models (the Rashomon set). By optimizing over an ellipsoidal approximation of this set, ElliCE ensures that the recommended recourse actions are stable even if the underlying model is retrained or updated.

## Installation

```bash
pip install .
```

## Features

*   **Provable Robustness**: Generates counterfactuals valid for *every* model in the approximated Rashomon set.
*   **Actionability Constraints**: Supports real-world constraints:
    *   **Immutable Features**: Freeze features that cannot be changed (e.g., `Age`, `Race`).
    *   **Range Constraints**: Restrict changes to feasible intervals (e.g., `Salary` within valid brackets).
    *   **One-Way Changes**: Enforce monotonic changes (e.g., `Experience` can only increase).
    *   **Categorical Features**: Handles one-hot encoded variables correctly using Gumbel-Softmax optimization.
*   **Dual Modes**:
    *   **Continuous**: Gradient-based optimization for finding new, optimal counterfactuals.
    *   **Discrete (Data-Supported)**: Selects the best robust candidates from existing data points.
*   **Backend Support**: Works seamlessly with both **Scikit-Learn** (Logistic Regression) and **PyTorch** models.

## Quick Start

```python
import ellice
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load Data & Train Model
data_raw = load_breast_cancer()
X = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
y = pd.Series(data_raw.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=5000).fit(X_train, y_train)

# 2. Initialize ElliCE
full_df = X_train.copy()
full_df['target'] = y_train
data = ellice.Data(dataframe=full_df, target_column='target')

exp = ellice.Explainer(
    model=clf,
    data=data,
    backend='sklearn'
)

# 3. Generate Robust Counterfactual
query = X_test.iloc[0]
target_class = 1 - clf.predict([query])[0]

cf = exp.generate_counterfactuals(
    query_instances=query,
    method='continuous',
    target_class=target_class,
    robustness_epsilon=0.01,  # Tolerance for Rashomon set size
    features_to_vary='all',   # Or list of specific features
    return_probs=True
)

print(cf)
```

## Advanced Usage: Actionability Constraints

ElliCE allows you to specify detailed constraints to ensure the generated recourse is practical.

### 1. Handling Categorical Features (One-Hot Encoding)

If your data contains one-hot encoded features, group them so ElliCE treats them as a single categorical variable.

```python
# Example: 'cat_Low', 'cat_Medium', 'cat_High' are one-hot encoded columns
one_hot_groups = [['cat_Low', 'cat_Medium', 'cat_High']]

cf = exp.generate_counterfactuals(
    query, 
    method='continuous',
    one_hot_groups=one_hot_groups,
    ...
)
```

### 2. Immutable Features

Prevent specific features from changing.

```python
# "Age" and "Race" will remain fixed
cf = exp.generate_counterfactuals(
    query,
    method='continuous',
    features_to_vary=[col for col in X.columns if col not in ['Age', 'Race']],
    ...
)
```

### 3. Range Constraints & One-Way Changes

Restrict the direction or magnitude of changes.

```python
# Salary must be between 30k and 100k
ranges = {'Salary': [30000, 100000]}

# Age can only increase (monotonicity)
one_way = {'Age': 'increase'}

cf = exp.generate_counterfactuals(
    query,
    method='continuous',
    permitted_range=ranges,
    one_way_change=one_way,
    ...
)
```

## Generators

### Continuous Generator (`method='continuous'`)
Optimizes the input features directly using gradient descent.
*   **Pros**: Finds the mathematically optimal counterfactual closest to the query.
*   **Cons**: May produce synthetic points that don't exist in the data (though usually plausible).
*   **Best for**: Numerical data, or when flexibility is key.

### Discrete Generator (`method='discrete'`)
Selects the best counterfactual from the existing training data (or a provided candidate set).
*   **Pros**: Guarantees the counterfactual is a real, observed data point (high plausibility).
*   **Cons**: Limited by the availability of data points; may not find a solution if the dataset is sparse.
*   **Best for**: Highly constrained domains (e.g., medical) where synthetic examples are risky.

## Citation

If you use ElliCE in your research, please cite:

```bibtex
@inproceedings{turbal2024ellice,
  title={ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets},
  author={Turbal, Bohdan and Voitsitska, Iryna and Semenova, Lesia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```
