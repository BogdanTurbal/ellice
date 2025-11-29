from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree, BallTree

from .base import EllipsoidGenerator

class DataSupportedGenerator(EllipsoidGenerator):
    """
    Data-supported (Discrete) Counterfactual Generator.
    Selects the best counterfactual from existing data points (candidates)
    that satisfy robustness and actionability constraints.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidates = None
        
    def _precompute_candidates(self, target_class: int = 1):
        """
        Identify all robust candidates from the training data.
        """
        if self._candidates is not None:
            # Assuming target class doesn't change, or we re-compute if needed.
            # For simplicity, we recompute if logic is cheap or cache it properly.
            # The definition of "Robust" depends on target_class. 
            # If we computed for class 1, it might not be valid for class 0.
            pass
            
        # Always recompute or check cache. 
        # Since this is fast enough for moderate data, let's recompute or check a flag.
        
        # Use training data as support set
        df = self.data.get_dev_data()
        X_support = df.values.astype(np.float32)
        
        batch_size = 1024
        robust_indices = []
        
        X_tensor_all = torch.from_numpy(X_support).to(self.device, self.dtype)
        
        with torch.no_grad():
            for i in range(0, len(X_support), batch_size):
                batch = X_tensor_all[i:i+batch_size]
                
                h_flat = self._get_penult_features(batch)
                bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
                h_aug = torch.cat([h_flat, bias], dim=1)
                
                inv_sqrt = self.Q_inv_sqrt
                u = inv_sqrt @ h_aug.T 
                norms = u.norm(dim=0, keepdim=True)
                u_norm = u / (norms + 1e-9)
                direction = (inv_sqrt @ u_norm).T
                
                term1 = torch.matmul(h_aug, self.omega_c).squeeze()
                term2 = (h_aug * direction).sum(dim=1) 
                
                # term2 calculation above is approximate to original code logic
                # Let's use exact formula: robust_logit = term1 - ||Q^{-1/2} h||
                # term2 above is effectively ||Q^{-1/2} h||?
                # direction = Q^{-1/2} (Q^{-1/2} h / ||...||)
                # h^T direction = h^T Q^{-1} h / ||...||
                # This matches logic if computed correctly. 
                # Simpler: term2 = norms.squeeze()
                
                term2_exact = norms.squeeze()
                
                if target_class == 1:
                    robust_logits = term1 - term2_exact
                    is_robust = robust_logits > 0
                else:
                    robust_logits = term1 + term2_exact
                    is_robust = robust_logits < 0
                
                robust_indices.extend((is_robust.cpu().numpy()).nonzero()[0] + i)
                
        self._candidates = df.iloc[robust_indices].reset_index(drop=True)

    def generate(
        self, 
        query_instance: pd.Series, 
        k: int = 1,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[float]]] = None,
        one_way_change: Optional[Dict[str, str]] = None,
        target_class: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        
        self._precompute_candidates(target_class)
        
        if self._candidates is None or self._candidates.empty:
            print("No robust candidates found in the data support set.")
            return pd.DataFrame()
            
        valid_candidates = self._candidates.copy()
        
        # 1. Immutable Features
        if features_to_vary is not None and features_to_vary != 'all':
             immutable = set(self.data.feature_names) - set(features_to_vary)
             for feat in immutable:
                 val = query_instance[feat]
                 valid_candidates = valid_candidates[np.isclose(valid_candidates[feat], val, atol=1e-5)]

        # 2. Permitted Ranges
        if permitted_range:
            for feat, (min_v, max_v) in permitted_range.items():
                valid_candidates = valid_candidates[
                    (valid_candidates[feat] >= min_v) & (valid_candidates[feat] <= max_v)
                ]
                
        # 3. One-way Changes
        if one_way_change:
            for feat, direction in one_way_change.items():
                if direction == 'increase':
                     valid_candidates = valid_candidates[valid_candidates[feat] >= query_instance[feat] - 1e-5]
                elif direction == 'decrease':
                     valid_candidates = valid_candidates[valid_candidates[feat] <= query_instance[feat] + 1e-5]

        if valid_candidates.empty:
            print("No candidates satisfy actionability constraints.")
            return pd.DataFrame()

        # Find nearest neighbors
        query_vals = query_instance.values
        dists = np.abs(valid_candidates.values - query_vals).sum(axis=1)
        
        sorted_idx = np.argsort(dists)
        best_indices = sorted_idx[:k]
        
        return valid_candidates.iloc[best_indices]
