from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree, BallTree

from .base import EllipsoidGenerator
from ..configs import AlgorithmConfig

class DataSupportedGenerator(EllipsoidGenerator):
    """
    Data-supported (Discrete) Counterfactual Generator.
    Selects the best counterfactual from existing data points (candidates)
    that satisfy robustness and actionability constraints.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidates = {}
        
    def _precompute_candidates(self, target_class: int = 1):
        """
        Identify all robust candidates from the training data.
        """
        if target_class in self._candidates:
            return
        
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
                # Add epsilon to avoid division by zero
                u_norm = u / (norms + 1e-9)
                direction = (inv_sqrt @ u_norm).T
                
                term1 = torch.matmul(h_aug, self.omega_c).squeeze()
                
                # Exact calculation: robust_logit = term1 - ||Q^{-1/2} h||
                # norms already holds ||Q^{-1/2} h||
                
                term2_exact = norms.squeeze()
                
                if target_class == 1:
                    robust_logits = term1 - term2_exact
                    is_robust = robust_logits > 0
                else:
                    robust_logits = term1 + term2_exact
                    is_robust = robust_logits < 0
                
                robust_indices.extend((is_robust.cpu().numpy()).nonzero()[0] + i)
                
        self._candidates[target_class] = df.iloc[robust_indices].reset_index(drop=True)

    def generate(
        self, 
        query_instance: pd.Series, 
        k: int = 1,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[float]]] = None,
        one_way_change: Optional[Dict[str, str]] = None,
        target_class: int = 1,
        search_mode: str = 'filtering', # 'filtering', 'kdtree', 'ball_tree'
        sparsity: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        
        # Validate search_mode and sparsity combination FIRST
        if search_mode == 'filtering':
            pass # Valid for both sparsity=True/False
        elif search_mode == 'kdtree':
            if sparsity:
                print(f"Error generating CF for index {kwargs.get('original_index', 'unknown')}: search_mode='kdtree' does not support sparsity=True. Use 'ball_tree' or 'filtering'.")
                raise ValueError("search_mode='kdtree' does not support sparsity=True. Use 'ball_tree' or 'filtering'.")
        elif search_mode == 'ball_tree':
            if not sparsity:
                print(f"Error generating CF for index {kwargs.get('original_index', 'unknown')}: search_mode='ball_tree' requires sparsity=True.")
                raise ValueError("search_mode='ball_tree' requires sparsity=True.")
        else:
            raise ValueError(f"Unknown search_mode: {search_mode}")
        
        self._precompute_candidates(target_class)
        candidates_df = self._candidates[target_class]
        
        if candidates_df is None or candidates_df.empty:
            print("No robust candidates found in the data support set.")
            return pd.DataFrame()
            
        valid_candidates = candidates_df.copy()
        
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

        # If no restrictions are specified (or even if they are), use search mode logic
        # The logic above already filters candidates. 
        # If search_mode is 'kdtree' or 'ball_tree', we build the index on valid_candidates.
        
        query_vals = query_instance.values.reshape(1, -1)
        candidate_vals = valid_candidates.values
        # Validation moved to start of method
        
        if search_mode == 'kdtree' and not sparsity:
            # Standard mode: K-d Tree on L2 distance
            # Only works if features are reasonable for L2 (standardized)
            tree = KDTree(candidate_vals)
            dist, ind = tree.query(query_vals, k=min(k, len(valid_candidates)))
            best_indices = ind[0]
            
        elif search_mode == 'ball_tree' and sparsity:
            # Sparsity mode: Ball Tree with custom metric
            # d(x, y) = C * Hamming(x, y) + L1(x, y)
            
            # Note: We prefer using sklearn's BallTree with a custom metric for research fidelity,
            # even if it's slower in pure Python than brute force.
            
            C = AlgorithmConfig.sparsity_constant
            
            def sparsity_metric(x, y):
                # x, y are 1D arrays
                diffs = np.abs(x - y)
                # Use tolerance for float equality
                hamming = (diffs > 1e-5).sum()
                l1 = diffs.sum()
                return C * hamming + l1

            # Build BallTree with custom metric
            # Note: This might be slow for large datasets
            tree = BallTree(candidate_vals, metric='pyfunc', func=sparsity_metric)
            dist, ind = tree.query(query_vals, k=min(k, len(valid_candidates)))
            best_indices = ind[0]
            
        elif sparsity:
             # Sparsity requested but search_mode='filtering' (default)
             # Use brute force with sparsity metric
            C = AlgorithmConfig.sparsity_constant
            diffs = np.abs(candidate_vals - query_vals)
            hamming = (diffs > 1e-5).sum(axis=1)
            l1 = diffs.sum(axis=1)
            scores = C * hamming + l1
            sorted_idx = np.argsort(scores)
            best_indices = sorted_idx[:k]

        else:
            # Default: Filtering / Brute Force L2 (Standard Mode)
            dists = np.sqrt(np.sum((candidate_vals - query_vals)**2, axis=1))
            sorted_idx = np.argsort(dists)
            best_indices = sorted_idx[:k]
        
        return valid_candidates.iloc[best_indices]
