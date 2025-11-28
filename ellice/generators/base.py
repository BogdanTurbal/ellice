import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from ..data import Data
from ..models.wrappers import ModelWrapper
from ..utils.helpers import compute_ellipsoid, safe_log1pexp

class EllipsoidGenerator(ABC):
    """
    Base class for Ellipsoid-based Counterfactual Generators.
    Manages the Rashomon set approximation.
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        data: Data,
        eps: float = 0.01,
        reg_coef: float = 1e-4,
        device: str = "cpu"
    ):
        self.model_wrapper = model
        self.data = data
        self.eps = eps
        self.reg_coef = reg_coef
        self.device = torch.device(device)
        self.dtype = torch.float32
        
        # 1. Setup Model and Split
        self.torch_model = self.model_wrapper.get_torch_model().to(self.device, self.dtype).eval()
        self.penult_model, self.theta_star = self.model_wrapper.split_model()
        self.penult_model = self.penult_model.to(self.device)
        self.theta_star = self.theta_star.to(self.device)
        self.omega_c = self.theta_star.detach()
        
        # 2. Process Training Data (for Hessian/Ellipsoid)
        # We need the training data to compute the empirical Hessian
        X_train = self.data.get_dev_data().values.astype(np.float32)
        y_train = self.data.get_target_data().values.astype(np.float32)
        
        # 3. Compute Penultimate Features
        with torch.no_grad():
            H_flat = self._get_penult_features(X_train)
            # Add bias term
            bias = torch.ones(H_flat.size(0), 1, device=self.device, dtype=self.dtype)
            H_aug_train = torch.cat([H_flat, bias], dim=1)
            
        H_feats_train = H_aug_train.cpu().numpy()
        self.H_feats_train = H_feats_train # Save for validation/debugging
        
        # 4. Compute Ellipsoid
        self.Q, self.Q_inv_sqrt, self.L_star, self.theta_threshold = compute_ellipsoid(
            H_feats_train, 
            self.theta_star, 
            self.reg_coef, 
            self.eps, 
            y_train, 
            self.device
        )
        
        # 5. Feature bounds
        self.feature_mins = torch.tensor(X_train.min(axis=0), device=self.device, dtype=self.dtype)
        self.feature_maxs = torch.tensor(X_train.max(axis=0), device=self.device, dtype=self.dtype)

    def _get_penult_features(self, X):
        """Extract penultimate features."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, self.dtype)
        
        # If X is 1D, unsqueeze
        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        features = self.penult_model(X)
        return features.view(features.size(0), -1)

    def _compute_worst_model(self, h_aug: torch.Tensor) -> torch.Tensor:
        """
        Analytically computes the worst-case model in the ellipsoid for a given input.
        """
        with torch.no_grad():
            if h_aug.dim() > 1:
                h_aug = h_aug[0]
                
            inv_sqrt = self.Q_inv_sqrt
            u = inv_sqrt @ h_aug
            norm_u = u.norm()
            
            if norm_u < 1e-9:
                return self.omega_c
                
            direction = inv_sqrt @ u / norm_u
            worst_theta = self.omega_c - direction
            return worst_theta
            
    def _robust_logit(self, x_tensor: torch.Tensor) -> float:
        """Computes the robust logit (worst-case prediction) for an input."""
        with torch.no_grad():
            h_flat = self._get_penult_features(x_tensor)
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
            
            worst_theta = self._compute_worst_model(h_aug)
            return torch.matmul(h_aug, worst_theta).item()

    def get_model_prob(self, x_df: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of the positive class (1) using the original model.
        """
        return self.model_wrapper.predict_proba(x_df.values)[:, 1] # Assuming binary classification

    def get_worst_case_prob(self, x_df: pd.DataFrame, target_class: int = 1) -> np.ndarray:
        """
        Returns the probability of the positive class (1) using the worst-case model.
        If target_class is 1, worst-case means minimizing the probability of class 1.
        If target_class is 0, worst-case means maximizing the probability of class 1 (minimizing class 0).
        
        Actually, to be consistent, this function returns the worst-case probability OF THE TARGET CLASS.
        """
        X_tensor = torch.tensor(x_df.values, dtype=self.dtype, device=self.device)
        probs = []
        
        with torch.no_grad():
            for i in range(X_tensor.shape[0]):
                # Compute features
                h_flat = self._get_penult_features(X_tensor[i:i+1])
                bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
                h_aug = torch.cat([h_flat, bias], dim=1)
                
                # Compute robust logit
                # logit_min = center - norm
                # logit_max = center + norm
                
                inv_sqrt = self.Q_inv_sqrt
                u = inv_sqrt @ h_aug.T
                norm = u.norm()
                center_logit = torch.matmul(h_aug, self.omega_c).item()
                
                if target_class == 1:
                    # We want Class 1. Worst case is minimal logit.
                    worst_logit = center_logit - norm.item()
                    prob = 1 / (1 + np.exp(-worst_logit))
                else:
                    # We want Class 0. Worst case for Class 0 is maximal logit (high prob for Class 1).
                    # Or do we return Prob(Class 0)?
                    # Let's return Prob(Target Class).
                    # Worst case for Class 0 is minimal Prob(Class 0) => Maximal Prob(Class 1).
                    worst_logit_for_class_1 = center_logit + norm.item()
                    prob_class_1 = 1 / (1 + np.exp(-worst_logit_for_class_1))
                    prob = 1 - prob_class_1
                    
                probs.append(prob)
                
        return np.array(probs)

    @abstractmethod
    def generate(self, query_instance: pd.Series, **kwargs) -> pd.DataFrame:
        pass

