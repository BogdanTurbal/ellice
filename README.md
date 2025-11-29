[![Pypi](https://img.shields.io/pypi/v/nanogcg?color=blue)](https://pypi.org/project/ellice)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS_2025-Spotlight-red)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118970)

# ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets

**ellice** is an efficient library for generating provably robust counterfactual explanations using ElliCE method. It ensures that recommended recourse actions remain valid across the set of all nearly-optimal models (the Rashomon set) using an ellipsoidal approximation, providing stability even if the underlying model is retrained or updated.

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Advanced Usage: Actionability Constraints](#advanced-usage-actionability-constraints
)
- [Generators](#generators)
- [Custom Backend Models](#custom-backend-models)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Installation

```bash
pip install ellice
```

Or from github:

```bash
git clone https://github.com/BogdanTurbal/ellice.git
cd ellice
pip install -e .
```

## Features

*   **Provable Robustness**: Generates counterfactuals valid for *every* model in the approximated Rashomon set.
*   **Actionability Constraints**: Supports real-world constraints:
    *   **Immutable Features**: Freeze features that cannot be changed (e.g., `Age`, `Race`).
    *   **Range Constraints**: Restrict changes to feasible intervals (e.g., `Salary` within valid brackets).
    *   **One-Way Changes**: Enforce monotonic changes (e.g., `Experience` can only increase).
    *   **Allowed Values**: Restrict ordinal/discrete features to specific valid values (e.g., `Education Level` $\in \{1, 2, 3, 4\}$).
    *   **Categorical Features**: Handles one-hot encoded variables correctly using Gumbel-Softmax optimization.
*   **Dual Modes**:
    *   **Continuous**: Gradient-based optimization for finding new, optimal counterfactuals.
    *   **Data-Supported**: Selects the best robust candidates from existing data points.
    *   **Sparsity Support**: Find counterfactuals with minimal feature changes (available for both modes).
*   **Backend Support**: Works seamlessly with both **Scikit-Learn** (Logistic Regression) and **PyTorch** models.
*   **Device Agnostic**: Automatic GPU/CPU detection with explicit device control (CUDA/CPU).
*   **Deterministic Execution**: Reproducible results with proper random seeding.

## Configuration

ElliCE uses a robust configuration system for advanced control. The configuration is split into two classes:

### GenerationConfig
Controls default parameters for counterfactual generation:
- `learning_rate`: Optimization learning rate (default: 0.1)
- `max_iterations`: Maximum optimization iterations (default: 1000)
- `patience`: Early stopping patience (default: 50)
- `robustness_weight`: Weight for robustness loss (default: 1.0)
- `proximity_weight`: Weight for proximity loss (default: 0.0)
- `gumbel_temperature`: Temperature for Gumbel-Softmax (default: 1.0)
- `early_stopping`: Enable early stopping (default: True)
- `progress_bar`: Show progress bar (default: True)

### AlgorithmConfig
Controls algorithmic stability and internal constants:
- `epsilon`: Numerical stability epsilon (default: 1e-9)
- `clip_grad_norm`: Gradient clipping threshold (default: 1.0)
- `gumbel_epsilon`: Gumbel-Softmax epsilon (default: 1e-10)
- `sparsity_constant`: Constant C in sparsity metric (C × Hamming + L1) (default: 100.0)
- `device`: Device selection - "auto" (default), "cpu" or "cuda"

```python
from ellice.ellice.configs import GenerationConfig, AlgorithmConfig

# Customize default behavior globally if needed
GenerationConfig.patience = 100
GenerationConfig.learning_rate = 0.05
AlgorithmConfig.epsilon = 1e-8
AlgorithmConfig.device = "cuda"  # Force CUDA, or use "auto" for automatic detection
```

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
    backend='sklearn',
    device='auto'  # Automatically selects CUDA if available, else CPU
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

### 4. Allowed Values (Ordinal Features)

Restrict specific features to a discrete set of valid values (e.g., integers for years of experience).

```python
# 'Education' can only be 1, 2, 3, or 4
allowed = {'Education': [1.0, 2.0, 3.0, 4.0]}

cf = exp.generate_counterfactuals(
    query,
    method='continuous',
    allowed_values=allowed,
    ...
)
```

### 5. Custom Weighting

You can assign different importance weights to features or groups in the proximity loss function.

```python
# Make changing 'Salary' twice as expensive
feature_weights = {'Salary': 2.0}

cf = exp.generate_counterfactuals(
    query,
    method='continuous',
    feature_weights=feature_weights,
    ...
)
```

## Generators

### Continuous Generator (`method='continuous'`)
Optimizes the input features directly using gradient descent.
*   **Pros**: Finds the counterfactual closest to the query.
*   **Cons**: May produce synthetic points that don't exist in the data (though usually plausible).

**Sparsity Support**: Enable `sparsity=True` to find counterfactuals with minimal feature changes using iterative feature selection (Algorithm 3 from the paper).

```python
cf = exp.generate_counterfactuals(
    query,
    method='continuous',
    sparsity=True,  # Find sparse counterfactuals
    ...
)
```

### Data-Supported Generator (`method='data_supported'`)
Selects the best counterfactual from the existing training data (or a provided candidate set).
*   **Pros**: Guarantees the counterfactual is a real, observed data point (high plausibility).
*   **Cons**: Limited by the availability of data points; may not find a solution if the dataset is sparse.

**Search Modes**:
- `search_mode='filtering'` (default): Brute-force filtering of all candidates. Works with or without sparsity.
- `search_mode='kdtree'`: Fast KDTree-based nearest neighbor search. Only available when `sparsity=False` and no actionability restrictions.
- `search_mode='ball_tree'`: BallTree with custom sparsity-aware distance metric (C × Hamming + L1). Requires `sparsity=True` and no actionability restrictions.

```python
# Default filtering (always works)
cf = exp.generate_counterfactuals(
    query,
    method='data_supported',
    search_mode='filtering',
    sparsity=False,
    ...
)

# Fast KDTree (no sparsity, no restrictions)
cf = exp.generate_counterfactuals(
    query,
    method='data_supported',
    search_mode='kdtree',
    sparsity=False,
    ...
)

# Sparse search with BallTree (sparsity=True, no restrictions)
cf = exp.generate_counterfactuals(
    query,
    method='data_supported',
    search_mode='ball_tree',
    sparsity=True,
    ...
)
```

## Custom Backend Models

ElliCE supports custom model wrappers for complex architectures or non-standard models. To use a custom backend:

### 1. Create a Custom ModelWrapper

Your custom class must inherit from `ModelWrapper` and implement the required abstract methods:

```python
from ellice.ellice.models.wrappers import ModelWrapper
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class CustomModelWrapper(ModelWrapper):
    """Custom wrapper for your specific model architecture."""
    
    def __init__(self, model):
        super().__init__(model, backend='custom')
        self.model.eval()  # Set to evaluation mode
    
    def get_torch_model(self) -> nn.Module:
        """Return the underlying PyTorch model."""
        return self.model
    
    def split_model(self) -> Tuple[nn.Module, torch.Tensor]:
        """
        Split model into penultimate feature extractor and last layer.
        
        Returns:
            penult: nn.Module that extracts penultimate features
            theta: torch.Tensor of flattened (weights, bias) from last layer
        """
        # Example: For a model with structure [features -> hidden -> output]
        # Extract everything except the last layer as penult
        penult = nn.Sequential(*list(self.model.children())[:-1])
        
        # Get last layer parameters
        last_layer = list(self.model.children())[-1]
        weight = last_layer.weight.detach().view(-1)
        bias = last_layer.bias.detach()
        theta = torch.cat([weight, bias])
        
        return penult, theta
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions for input X."""
        device = next(self.model.parameters()).device
        X_tensor = torch.from_numpy(X).float().to(device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            if logits.shape[1] == 1:
                # Binary classification
                probs_1 = torch.sigmoid(logits)
                probs_0 = 1 - probs_1
                probs = torch.cat([probs_0, probs_1], dim=1)
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
```

### 2. Use Custom Backend in Explainer

```python
# Initialize with custom backend
exp = ellice.Explainer(
    model=your_custom_model,
    data=data,
    backend='custom',
    backend_model_class=CustomModelWrapper
)

# Generate counterfactuals as usual
cf = exp.generate_counterfactuals(query, ...)
```

### Important Notes

- **Model Structure**: Your model must have a linear last layer (for binary classification, output size 1; for multi-class, output size = number of classes).
- **Penultimate Features**: The `split_model()` method must correctly identify the penultimate layer. For complex architectures (e.g., ResNet, Transformer), you may need custom logic.
- **Device Handling**: Ensure your wrapper handles device placement correctly (CPU/CUDA/MPS).
- **Binary vs Multi-class**: The `predict_proba` method should handle both binary (1 output) and multi-class (N outputs) cases.

### Example: Complex Architecture

For models with non-sequential structures, you might need to use hooks or custom forward passes:

```python
class ComplexModelWrapper(ModelWrapper):
    def split_model(self) -> Tuple[nn.Module, torch.Tensor]:
        # For a model with skip connections or complex structure,
        # you might need to create a custom feature extractor
        class FeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.backbone = original_model.backbone
                self.hidden = original_model.hidden_layers
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.hidden(x)
                return x
        
        penult = FeatureExtractor(self.model)
        last_layer = self.model.classifier  # Assuming classifier is the last layer
        weight = last_layer.weight.detach().view(-1)
        bias = last_layer.bias.detach()
        theta = torch.cat([weight, bias])
        
        return penult, theta
```

## Reproducibility

For deterministic results, set random seeds before running:

```python
import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)  # Set before training models and generating counterfactuals
```

## Citation

If you use ElliCE in your research, please cite:

```bibtex
@inproceedings{turbal2025ellice,
  title={ElliCE: Efficient and Provably Robust Algorithmic Recourse via the Rashomon Sets},
  author={Turbal, Bohdan and Voitsitska, Iryna and Semenova, Lesia},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
