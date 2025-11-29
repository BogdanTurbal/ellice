import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, model, backend='pytorch'):
        self.model = model
        self.backend = backend

    @abstractmethod
    def get_torch_model(self) -> nn.Module:
        """Returns the underlying PyTorch model."""
        pass

    @abstractmethod
    def split_model(self) -> Tuple[nn.Module, torch.Tensor]:
        """
        Splits the model into penultimate feature extractor and last layer parameters.
        Returns:
            penult: nn.Module (feature extractor)
            theta: torch.Tensor (flattened weights+bias of last layer)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

class PyTorchModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model, backend='pytorch')
        self.model.eval()

    def get_torch_model(self) -> nn.Module:
        return self.model

    def split_model(self) -> Tuple[nn.Module, torch.Tensor]:
        """
        Splits a Sequential PyTorch model.
        Assumes the last layer is nn.Linear.
        """
        # Handle DataParallel or DistributedDataParallel
        model = self.model
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        # Find last linear layer
        last_linear = None
        modules = list(model.modules())
        for m in reversed(modules):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        
        if last_linear is None:
            raise ValueError("No nn.Linear layer found in the model.")

        # Construct penultimate model
        # This logic assumes a Sequential model structure. 
        # For complex architectures, user might need to provide the split manually.
        # We attempt to slice model.children()
        children = list(model.children())
        
        # Heuristic: assume last child is the linear layer or contains it
        # If the model is strictly sequential:
        if isinstance(model, nn.Sequential):
             penult = nn.Sequential(*children[:-1])
        else:
            # If not sequential, we might need a hook or a custom forward pass.
            # For this implementation, we'll try to clone the model and remove the last layer 
            # OR warn the user. 
            # A safer robust approach for arbitrary models is asking the user to define the split,
            # but for "Sequential" style MLPs it works.
            # We will fallback to specific robustx logic:
            # robustx implementation: penult = nn.Sequential(*children[:-2]) ?? 
            # Let's stick to the provided code's logic if possible or improve it.
            # The provided code did: penult = nn.Sequential(*children[:-2]) which implies specific structure (dropout etc).
            # Let's try to be more generic: remove the last layer.
             penult = nn.Sequential(*list(model.children())[:-1])

        # Flatten last layer
        weight = last_linear.weight.detach().view(-1)
        bias = last_linear.bias.detach()
        theta = torch.cat([weight, bias])
        
        return penult, theta

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            # Automatically detect device of the model parameters
            device = next(self.model.parameters()).device
            X = torch.from_numpy(X).float().to(device)
        
        with torch.no_grad():
            logits = self.model(X)
            if logits.shape[1] == 1:
                # Binary case: logits -> sigmoid -> probs for class 1
                probs_1 = torch.sigmoid(logits)
                probs_0 = 1 - probs_1
                probs = torch.cat([probs_0, probs_1], dim=1)
            else:
                # Multiclass case
                probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

class SklearnModel(ModelWrapper):
    """
    Wrapper for Sklearn models (LogisticRegression, LinearSVC, MLPClassifier).
    Converts them to a PyTorch equivalent for ElliCE processing.
    """
    def __init__(self, model):
        super().__init__(model, backend='sklearn')
        self.torch_model = self._convert_to_torch()

    def _convert_to_torch(self) -> nn.Module:
        from sklearn.linear_model import LogisticRegression
        
        if isinstance(self.model, LogisticRegression):
            input_dim = self.model.coef_.shape[1]
            torch_model = nn.Sequential(
                nn.Linear(input_dim, 1)
            )
            with torch.no_grad():
                torch_model[0].weight.copy_(torch.from_numpy(self.model.coef_))
                torch_model[0].bias.copy_(torch.from_numpy(self.model.intercept_))
            return torch_model
            
        # TODO: Add MLPClassifier support
        raise NotImplementedError("Only LogisticRegression is currently supported for Sklearn backend.")

    def get_torch_model(self) -> nn.Module:
        return self.torch_model

    def split_model(self) -> Tuple[nn.Module, torch.Tensor]:
        # For a single layer model (Logistic Regression), penultimate features are just inputs (Identity)
        # And the "last layer" is the model itself.
        
        class Identity(nn.Module):
            def forward(self, x): return x
            
        penult = Identity()
        
        # Last layer is the only layer
        last_linear = self.torch_model[0]
        weight = last_linear.weight.detach().view(-1)
        bias = last_linear.bias.detach()
        theta = torch.cat([weight, bias])
        
        return penult, theta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba (e.g. SVC without probability=True)
            d = self.model.decision_function(X)
            return 1 / (1 + np.exp(-d))

def load_model(model, backend='auto') -> ModelWrapper:
    if backend == 'auto':
        if isinstance(model, nn.Module):
            return PyTorchModel(model)
        else:
            return SklearnModel(model)
    elif backend == 'pytorch':
        return PyTorchModel(model)
    elif backend == 'sklearn':
        return SklearnModel(model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

