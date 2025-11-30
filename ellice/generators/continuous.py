import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import time

from tqdm import tqdm

from ..configs import AlgorithmConfig, GenerationConfig
from .base import EllipsoidGenerator

class ContinuousGenerator(EllipsoidGenerator):
    """
    Gradient-based Counterfactual Generator.
    Optimizes the input features directly to find a robust counterfactual.
    """
    
    def _gumbel_softmax_sample(self, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + AlgorithmConfig.gumbel_epsilon) + AlgorithmConfig.gumbel_epsilon)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def _initialize_optimization_setup(
        self,
        query_instance: pd.Series,
        feature_weights: Optional[Dict[str, float]],
        one_hot_groups: Optional[List[List[str]]],
        learning_rate: float,
        features_to_vary: Optional[List[str]]
    ):
        # Prepare Input
        x_orig = torch.tensor(query_instance.values, dtype=self.dtype, device=self.device).unsqueeze(0)
        feature_names = self.data.feature_names
        input_dim = len(feature_names)
        
        # Compute Weights Vector (for L2 loss)
        loss_weights = torch.ones(input_dim, device=self.device, dtype=self.dtype)
        
        # 1. Apply feature_weights
        if feature_weights:
            for feat, w in feature_weights.items():
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    loss_weights[idx] = w
                    
        # 2. Apply default 0.5 weight for one-hot groups if not specified
        if one_hot_groups:
            for group in one_hot_groups:
                indices = [feature_names.index(col) for col in group if col in feature_names]
                for idx in indices:
                    feat_name = feature_names[idx]
                    if feature_weights is None or feat_name not in feature_weights:
                        loss_weights[idx] = GenerationConfig.default_group_weight
        
        # Handle One-Hot Groups
        if one_hot_groups:
            one_hot_indices = []
            categorical_mask = np.zeros(input_dim, dtype=bool)
            
            for group in one_hot_groups:
                indices = [feature_names.index(col) for col in group if col in feature_names]
                if indices:
                    one_hot_indices.append(indices)
                    categorical_mask[indices] = True
            
            continuous_indices = [i for i in range(input_dim) if not categorical_mask[i]]
        else:
            one_hot_indices = []
            continuous_indices = list(range(input_dim))

        # Initialize Optimization Variables
        params = []
        
        # 1. Continuous Features
        if continuous_indices:
            x_cont = x_orig[0, continuous_indices].clone().detach().requires_grad_(True)
            params.append(x_cont)
        else:
            x_cont = None
            
        # 2. Categorical Logits
        cat_logits_list = []
        if one_hot_indices:
            for group_indices in one_hot_indices:
                current_vals = x_orig[0, group_indices]
                active_idx = torch.argmax(current_vals).item()
                
                logits = torch.full((len(group_indices),), 0.1, device=self.device, dtype=self.dtype, requires_grad=True)
                logits.data[active_idx] = 1.0 # Strong bias towards original
                
                cat_logits_list.append(logits)
                params.append(logits)
        
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Identify indices to freeze (Immutable features)
        frozen_indices_cont = [] # Indices relative to x_cont
        frozen_groups_cat = [] # Indices relative to cat_logits_list
        
        if features_to_vary is not None and features_to_vary != 'all':
            for i, global_idx in enumerate(continuous_indices):
                feat_name = feature_names[global_idx]
                if feat_name not in features_to_vary:
                    frozen_indices_cont.append(i)
            
            for i, group_indices in enumerate(one_hot_indices):
                group_names = [feature_names[idx] for idx in group_indices]
                if all(name not in features_to_vary for name in group_names):
                    frozen_groups_cat.append(i)
        elif features_to_vary == 'all':
            pass 
        elif GenerationConfig.features_to_vary:
             features_to_vary = GenerationConfig.features_to_vary
             for i, global_idx in enumerate(continuous_indices):
                feat_name = feature_names[global_idx]
                if feat_name not in features_to_vary:
                    frozen_indices_cont.append(i)
             for i, group_indices in enumerate(one_hot_indices):
                group_names = [feature_names[idx] for idx in group_indices]
                if all(name not in features_to_vary for name in group_names):
                    frozen_groups_cat.append(i)

        return {
            'x_orig': x_orig,
            'feature_names': feature_names,
            'input_dim': input_dim,
            'loss_weights': loss_weights,
            'one_hot_indices': one_hot_indices,
            'continuous_indices': continuous_indices,
            'x_cont': x_cont,
            'cat_logits_list': cat_logits_list,
            'params': params,
            'optimizer': optimizer,
            'frozen_indices_cont': frozen_indices_cont,
            'frozen_groups_cat': frozen_groups_cat
        }

    def _initialize_constraints(
        self,
        continuous_indices: List[int],
        permitted_range: Optional[Dict[str, List[float]]],
        allowed_values: Optional[Dict[str, List[float]]],
        one_way_change: Optional[Dict[str, str]]
    ):
        feature_names = self.data.feature_names
        
        # Merge Runtime and Config constraints
        if permitted_range is None and GenerationConfig.permitted_range:
            permitted_range = GenerationConfig.permitted_range
            
        if one_way_change is None and GenerationConfig.one_way_change:
            one_way_change = GenerationConfig.one_way_change

        if allowed_values is None and GenerationConfig.allowed_values:
            allowed_values = GenerationConfig.allowed_values
        
        range_constraints_cont = []
        allowed_values_cont = []
        one_way_constraints_cont = []

        if permitted_range:
            for feat, (min_v, max_v) in permitted_range.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        range_constraints_cont.append((local_idx, min_v, max_v))
        
        if allowed_values:
            for feat, vals in allowed_values.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        vals_sorted = torch.tensor(sorted(vals), device=self.device, dtype=self.dtype)
                        allowed_values_cont.append((local_idx, vals_sorted))
                    
        if one_way_change:
            for feat, direction in one_way_change.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        one_way_constraints_cont.append((local_idx, direction))
                        
        return range_constraints_cont, allowed_values_cont, one_way_constraints_cont

    def _project_continuous_features(
        self,
        x_cont: torch.Tensor,
        x_orig: torch.Tensor,
        continuous_indices: List[int],
        frozen_indices_cont: List[int],
        allowed_values_cont: List,
        range_constraints_cont: List,
        one_way_constraints_cont: List
    ):
        with torch.no_grad():
            if x_cont is not None:
                # 1. Enforce min/max of allowed values as global bounds
                for idx, vals in allowed_values_cont:
                    min_v, max_v = vals[0], vals[-1]
                    x_cont.data[idx] = torch.clamp(x_cont.data[idx], min_v, max_v)

                # Reset immutable features
                if frozen_indices_cont:
                    x_cont.data[frozen_indices_cont] = x_orig.data[0, continuous_indices][frozen_indices_cont]
                    
                # Clip to global bounds
                global_mins = self.feature_mins[continuous_indices]
                global_maxs = self.feature_maxs[continuous_indices]
                x_cont.data = torch.max(torch.min(x_cont.data, global_maxs), global_mins)
                
                # Clip to permitted ranges
                for idx, min_v, max_v in range_constraints_cont:
                    x_cont.data[idx] = torch.clamp(x_cont.data[idx], min_v, max_v)
                    
                # Enforce one-way changes
                for idx, direction in one_way_constraints_cont:
                    orig_val = x_orig.data[0, continuous_indices][idx]
                    if direction == 'increase':
                         x_cont.data[idx] = torch.max(x_cont.data[idx], orig_val)
                    elif direction == 'decrease':
                         x_cont.data[idx] = torch.min(x_cont.data[idx], orig_val)
                         
                # Snap to allowed values (Hard constraint)
                for idx, vals in allowed_values_cont:
                    val = x_cont.data[idx]
                    diff = (vals - val).abs()
                    min_idx = torch.argmin(diff)
                    x_cont.data[idx] = vals[min_idx]

    def generate(
        self,
        query_instance: pd.Series,
        learning_rate: float = GenerationConfig.learning_rate,
        max_iterations: int = GenerationConfig.max_iterations,
        robustness_weight: float = GenerationConfig.robustness_weight,
        proximity_weight: float = GenerationConfig.proximity_weight,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[float]]] = None,
        one_way_change: Optional[Dict[str, str]] = None,
        allowed_values: Optional[Dict[str, List[float]]] = None,
        one_hot_groups: Optional[List[List[str]]] = None,
        gumbel_temperature: float = GenerationConfig.gumbel_temperature,
        target_class: int = 1,
        early_stopping: bool = GenerationConfig.early_stopping,
        patience: int = GenerationConfig.patience,
        progress_bar: bool = GenerationConfig.progress_bar,
        gradient_mode: str = GenerationConfig.gradient_mode, 
        feature_weights: Optional[Dict[str, float]] = None,
        group_weights: Optional[Dict[str, float]] = None,
        allowed_ranges_closest_grad: bool = GenerationConfig.allowed_ranges_closest_grad,
        requires: str = "valid",
        **kwargs
    ) -> pd.DataFrame:
        
        # Prepare Input
        x_orig = torch.tensor(query_instance.values, dtype=self.dtype, device=self.device).unsqueeze(0)
        feature_names = self.data.feature_names
        input_dim = len(feature_names)
        
        # Compute Weights Vector (for L2 loss)
        # Default: 1.0 for everything
        loss_weights = torch.ones(input_dim, device=self.device, dtype=self.dtype)
        
        # 1. Apply feature_weights
        if feature_weights:
            for feat, w in feature_weights.items():
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    loss_weights[idx] = w
                    
        # 2. Apply default 0.5 weight for one-hot groups if not specified
        if one_hot_groups:
            for group in one_hot_groups:
                
                indices = [feature_names.index(col) for col in group if col in feature_names]
                for idx in indices:
                    feat_name = feature_names[idx]
                    # If not explicitly overridden by feature_weights
                    if feature_weights is None or feat_name not in feature_weights:
                        loss_weights[idx] = GenerationConfig.default_group_weight
        
        # Handle One-Hot Groups
        if one_hot_groups:
            # Map column names to indices
            one_hot_indices = []
            categorical_mask = np.zeros(input_dim, dtype=bool)
            
            for group in one_hot_groups:
                indices = [feature_names.index(col) for col in group if col in feature_names]
                if indices:
                    one_hot_indices.append(indices)
                    categorical_mask[indices] = True
            
            continuous_indices = [i for i in range(input_dim) if not categorical_mask[i]]
        else:
            one_hot_indices = []
            continuous_indices = list(range(input_dim))

        # Initialize Optimization Variables
        params = []
        
        # 1. Continuous Features
        if continuous_indices:
            x_cont = x_orig[0, continuous_indices].clone().detach().requires_grad_(True)
            params.append(x_cont)
        else:
            x_cont = None
            
        # 2. Categorical Logits
        cat_logits_list = []
        if one_hot_indices:
            for group_indices in one_hot_indices:
                # Initialize logits to favor the current category strongly
                current_vals = x_orig[0, group_indices]
                active_idx = torch.argmax(current_vals).item()
                
                # Initialize with 0.1, then set active to 2.0
                logits = torch.full((len(group_indices),), 0.1, device=self.device, dtype=self.dtype, requires_grad=True)
                logits.data[active_idx] = 1.0 # Strong bias towards original
                
                cat_logits_list.append(logits)
                params.append(logits)
        
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Target
        target = torch.tensor([float(target_class)], device=self.device, dtype=self.dtype)
        
        # Best tracking
        best_x = x_orig.clone().detach()
        best_robust_logit = -float('inf')
        best_val_term1 = None  # Track best original model logit for "valid" check
        
        # Early Stopping
        no_improve_count = 0
        
        # Identify indices to freeze (Immutable features)
        frozen_indices_cont = [] # Indices relative to x_cont
        frozen_groups_cat = [] # Indices relative to cat_logits_list
        
        if features_to_vary is not None and features_to_vary != 'all':
            # Check continuous features
            for i, global_idx in enumerate(continuous_indices):
                feat_name = feature_names[global_idx]
                if feat_name not in features_to_vary:
                    frozen_indices_cont.append(i)
            
            # Check categorical groups (if any feature in group is mutable, we assume the group is mutable)
            # Or typically, the whole group is either mutable or immutable.
            for i, group_indices in enumerate(one_hot_indices):
                group_names = [feature_names[idx] for idx in group_indices]
                # If ALL features in group are NOT in features_to_vary, then freeze
                if all(name not in features_to_vary for name in group_names):
                    frozen_groups_cat.append(i)
        elif features_to_vary == 'all':
            pass # Nothing frozen by default
        elif GenerationConfig.features_to_vary:
             # Use default config if provided
             features_to_vary = GenerationConfig.features_to_vary
             # Re-run logic (simplified by recursion or just duplicate for now to avoid complexity)
             for i, global_idx in enumerate(continuous_indices):
                feat_name = feature_names[global_idx]
                if feat_name not in features_to_vary:
                    frozen_indices_cont.append(i)
             for i, group_indices in enumerate(one_hot_indices):
                group_names = [feature_names[idx] for idx in group_indices]
                if all(name not in features_to_vary for name in group_names):
                    frozen_groups_cat.append(i)

        # Merge Runtime and Config constraints
        # If runtime is None, check config
        if permitted_range is None and GenerationConfig.permitted_range:
            permitted_range = GenerationConfig.permitted_range
            
        if one_way_change is None and GenerationConfig.one_way_change:
            one_way_change = GenerationConfig.one_way_change

        if allowed_values is None and GenerationConfig.allowed_values:
            allowed_values = GenerationConfig.allowed_values
        
        # Pre-compute indices for constraints to speed up loop
        range_constraints_cont = [] # (local_cont_idx, min, max)
        allowed_values_cont = [] # (local_cont_idx, sorted_tensor_values)

        if permitted_range:
            for feat, (min_v, max_v) in permitted_range.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        range_constraints_cont.append((local_idx, min_v, max_v))
        
        if allowed_values:
            for feat, vals in allowed_values.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        # Sort values and convert to tensor
                        vals_sorted = torch.tensor(sorted(vals), device=self.device, dtype=self.dtype)
                        allowed_values_cont.append((local_idx, vals_sorted))
                    
        one_way_constraints_cont = [] # (local_cont_idx, direction)
        if one_way_change:
            for feat, direction in one_way_change.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        one_way_constraints_cont.append((local_idx, direction))

        # Pre-compute constant
        inv_sqrt = self.Q_inv_sqrt
        
        # Setup Progress Bar
        iterator = range(max_iterations)
        #self.progress_bar = None
        if progress_bar:
            print("Progress Bar Enabled")
            iterator = tqdm(iterator, desc="Generating CF", leave=True, mininterval=0)
            #self.progress_bar = iterator

        for i in iterator:
            optimizer.zero_grad()
            
            # Reconstruct x_cf from components
            x_cf_full = torch.zeros_like(x_orig)
            
            if x_cont is not None:
                x_cf_full[0, continuous_indices] = x_cont
                
            if cat_logits_list:
                for group_idx, logits in enumerate(cat_logits_list):
                    if group_idx in frozen_groups_cat:
                        # Use original values if frozen
                        orig_vals = x_orig[0, one_hot_indices[group_idx]]
                        x_cf_full[0, one_hot_indices[group_idx]] = orig_vals
                    else:
                        # Sample
                        probs = self._gumbel_softmax_sample(logits, tau=gumbel_temperature)
                        x_cf_full[0, one_hot_indices[group_idx]] = probs
            
            # 1. Compute Features
            h_flat = self._get_penult_features(x_cf_full)
            bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
            h_aug = torch.cat([h_flat, bias], dim=1)
    
            
            # Always use Min-Max Strategy (Default)
            with torch.no_grad():
                worst_theta = self._compute_worst_model(h_aug)
            
            if target_class == 0:
                u = inv_sqrt @ h_aug.T 
                norm_u = u.norm()
                direction = inv_sqrt @ u / (norm_u + AlgorithmConfig.epsilon)
                worst_theta = self.omega_c + direction.squeeze()
                
            robust_logit = torch.matmul(h_aug, worst_theta)
            
            # Loss Logic
            if target_class == 1:
                loss_robust = F.relu( -robust_logit) 
            else:
                loss_robust = F.relu( robust_logit)

            # 3. Proximity Loss
            # Weighted L2
            diff = x_cf_full - x_orig
            
            weighted_diff_sq = loss_weights * (diff ** 2)
            loss_prox = torch.sqrt(torch.sum(weighted_diff_sq) + AlgorithmConfig.epsilon)
            
            # Total Loss
            loss = robustness_weight * loss_robust + proximity_weight * loss_prox
            
            loss.backward()
            
            # "Gradient Interpolation" for Allowed Values
            if x_cont is not None and x_cont.grad is not None and allowed_values_cont:
                
                # Construct batch
                batch_inputs = []
                interpolation_weights = [] # (idx, w_left, w_right)
                
                with torch.no_grad():
                    current_x = x_cf_full.clone()
                    
                    for idx, vals in allowed_values_cont:
                        val = x_cont.data[idx]
                        
                        # Find nearest
                        diff = val - vals
                        left_mask = diff >= 0
                        v_left = vals[left_mask].max() if left_mask.any() else vals[0]
                        right_mask = diff <= 0
                        v_right = vals[right_mask].min() if right_mask.any() else vals[-1]
                        
                        # If allowed_ranges_closest_grad is True, we only use the closest one
                        if allowed_ranges_closest_grad:
                            # Distance to left and right
                            d_left = (val - v_left).abs()
                            d_right = (v_right - val).abs()
                            
                            if d_left <= d_right:
                                v_target = v_left
                            else:
                                v_target = v_right
                                
                            interpolation_weights.append((idx, 1.0 if v_target == v_left else 0.0, 1.0 if v_target == v_right else 0.0))
                            
                            
                            x_left = current_x.clone()
                            x_left[0, continuous_indices[idx]] = v_left
                            batch_inputs.append(x_left)
                            
                            x_right = current_x.clone()
                            x_right[0, continuous_indices[idx]] = v_right
                            batch_inputs.append(x_right)
                            
                            continue

                        if v_left == v_right:
                            interpolation_weights.append((idx, 0.5, 0.5)) # Doesn't matter
                            # Add dummy duplicates to keep indexing consistent
                            batch_inputs.append(current_x)
                            batch_inputs.append(current_x)
                            continue
                            
                        d_left = (val - v_left).abs()
                        d_right = (v_right - val).abs()
                        total_d = d_left + d_right + AlgorithmConfig.epsilon
                        p_left = 1 - (d_left / total_d)
                        p_right = 1 - (d_right / total_d)
                        sum_p = p_left + p_right
                        p_left /= sum_p
                        p_right /= sum_p
                        
                        interpolation_weights.append((idx, p_left, p_right))
                        
                        # Left Input
                        x_left = current_x.clone()
                        x_left[0, continuous_indices[idx]] = v_left
                        batch_inputs.append(x_left)
                        
                        # Right Input
                        x_right = current_x.clone()
                        x_right[0, continuous_indices[idx]] = v_right
                        batch_inputs.append(x_right)
                
                if batch_inputs:
                    # Stack batch: [2*K, input_dim]
                    batch_tensor = torch.cat(batch_inputs, dim=0)
                    # Enable grad for input to get dLoss/dx
                    batch_tensor.requires_grad_(True)
                    
                    # Forward Pass (Batch)
                    h_flat_batch = self._get_penult_features(batch_tensor)
                    bias_batch = torch.ones(h_flat_batch.size(0), 1, device=self.device, dtype=self.dtype)
                    h_aug_batch = torch.cat([h_flat_batch, bias_batch], dim=1)
                    
                    # Compute Robust Logit (Batch)
                    if gradient_mode == "full_grad":
                        u_batch = inv_sqrt @ h_aug_batch.T
                        term2_batch = torch.norm(u_batch, p=2, dim=0)
                        term1_batch = torch.matmul(h_aug_batch, self.omega_c).squeeze()
                        robust_logit_batch = term1_batch - term2_batch
                        if target_class == 0:
                            robust_logit_batch = term1_batch + term2_batch
                    else:
                        # Min-Max approx for batch (independent worst case per sample)
                        # This re-computes worst theta for each
                        # For efficiency we can use the same logic
                        with torch.no_grad():
                            # Analytic worst case
                            u_b = inv_sqrt @ h_aug_batch.T
                            norms_b = u_b.norm(dim=0)
                            dirs_b = (inv_sqrt @ u_b).T / (norms_b.unsqueeze(1) + AlgorithmConfig.epsilon)
                            if target_class == 0:
                                worst_thetas = self.omega_c + dirs_b
                            else:
                                worst_thetas = self.omega_c - dirs_b
                        
                        # Logit
                        robust_logit_batch = (h_aug_batch * worst_thetas).sum(dim=1)

                    # Loss (Batch)
                    if target_class == 1:
                        loss_rob_batch = -robust_logit_batch
                    else:
                        loss_rob_batch = robust_logit_batch
                        
                    # Proximity (Batch) - Distance to x_orig
                    # x_orig is [1, dim], batch is [N, dim]
                    loss_prox_batch = torch.norm(batch_tensor - x_orig, p=2, dim=1)
                    
                    total_loss_batch = robustness_weight * loss_rob_batch + proximity_weight * loss_prox_batch
                    
                    # Backward (Batch)
                    # We want grad of total_loss_batch w.r.t batch_tensor
                    # sum() lets us compute all grads in one go
                    total_loss_batch.sum().backward()
                    
                    # Extract Gradients
                    batch_grads = batch_tensor.grad # [2*K, input_dim]
                    
                    # Update x_cont.grad
                    # For each constrained feature, replace its gradient with weighted avg of discrete grads
                    for k, (idx, p_left, p_right) in enumerate(interpolation_weights):
                        # Indices in batch: 2*k and 2*k+1
                        grad_left = batch_grads[2*k, continuous_indices[idx]]
                        grad_right = batch_grads[2*k+1, continuous_indices[idx]]
                        
                        new_grad = p_left * grad_left + p_right * grad_right
                        
                        # Assign to main gradient
                        x_cont.grad[idx] = new_grad

            # Clip gradients for stability (helps with unscaled data)
            torch.nn.utils.clip_grad_norm_(params, max_norm=AlgorithmConfig.clip_grad_norm)

            optimizer.step()
            
            # 4. Projections / Constraints (Only on Continuous Variables)
            with torch.no_grad():
                if x_cont is not None:
                    
                    # 1. Enforce min/max of allowed values as global bounds for that feature
                    for idx, vals in allowed_values_cont:
                        min_v, max_v = vals[0], vals[-1]
                        x_cont.data[idx] = torch.clamp(x_cont.data[idx], min_v, max_v)

                    # Reset immutable features
                    if frozen_indices_cont:
                        x_cont.data[frozen_indices_cont] = x_orig.data[0, continuous_indices][frozen_indices_cont]
                        
                    # Clip to global bounds (of continuous features)
                    global_mins = self.feature_mins[continuous_indices]
                    global_maxs = self.feature_maxs[continuous_indices]
                    x_cont.data = torch.max(torch.min(x_cont.data, global_maxs), global_mins)
                    
                    # Clip to permitted ranges
                    for idx, min_v, max_v in range_constraints_cont:
                        x_cont.data[idx] = torch.clamp(x_cont.data[idx], min_v, max_v)
                        
                    # Enforce one-way changes
                    for idx, direction in one_way_constraints_cont:
                        orig_val = x_orig.data[0, continuous_indices][idx]
                        if direction == 'increase':
                             x_cont.data[idx] = torch.max(x_cont.data[idx], orig_val)
                        elif direction == 'decrease':
                             x_cont.data[idx] = torch.min(x_cont.data[idx], orig_val)
                             
                    # Snap to allowed values (Hard constraint)
                    # This effectively makes it discrete optimization via gradient descent + projection
                    for idx, vals in allowed_values_cont:
                        val = x_cont.data[idx]
                        # Find nearest
                        diff = (vals - val).abs()
                        min_idx = torch.argmin(diff)
                        x_cont.data[idx] = vals[min_idx]
                
                # Check validity
                # Reconstruct discrete version for validation
                x_val_full = x_cf_full.clone()
                
                # Hard discretization for categorical
                if cat_logits_list:
                     for group_idx, logits in enumerate(cat_logits_list):
                        if group_idx not in frozen_groups_cat:
                            probs = F.softmax(logits, dim=0)
                            best_idx = torch.argmax(probs).item()
                            # Zero out group
                            indices = one_hot_indices[group_idx]
                            x_val_full[0, indices] = 0.0
                            x_val_full[0, indices[best_idx]] = 1.0
                            
                h_flat_val = self._get_penult_features(x_val_full)
                h_aug_val = torch.cat([h_flat_val, bias], dim=1)
                
                val_term1 = torch.matmul(h_aug_val, self.omega_c).item()
                val_term2 = torch.norm(inv_sqrt @ h_aug_val.T).item()
                
                if target_class == 1:
                    val_robust_logit = val_term1 - val_term2
                    current_prob = 1 / (1 + np.exp(-val_term1))
                    robust_prob = 1 / (1 + np.exp(-val_robust_logit))
                    metric = val_robust_logit
                else:
                    val_robust_logit = val_term1 + val_term2
                    current_prob = 1 - (1 / (1 + np.exp(-val_term1)))
                    robust_prob = 1 - (1 / (1 + np.exp(-val_robust_logit)))
                    metric = -val_robust_logit 

                # Update Best logic
                # We still track the max robust logit found so far, but if we cross 0, we might want to stop.
                if metric > best_robust_logit:
                    best_robust_logit = metric
                    best_x = x_val_full.clone().detach()
                    best_val_term1 = val_term1  # Store original model logit
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Update Progress Bar
                if progress_bar:
                    iterator.set_postfix({
                        'Prob': f"{current_prob:.3f}",
                        'RobLogit': f"{-val_robust_logit:.3f}",
                        'BestRobLogit': f"{best_robust_logit:.3f}"
                    })

                # Early Stopping
                # 1. Standard patience
                if early_stopping and no_improve_count >= patience:
                    import warnings
                    warnings.warn(
                        f"\n Early stopping due to failed convergence and no improvement for {patience} steps. "
                        f"Try to increase regularization_coefficient or decrease robustness_epsilon",
                        UserWarning
                    )
                    if progress_bar:
                        iterator.close()
                    break
                    
                # 2. Stop if robust (User request)
                if metric > 0:
                    if progress_bar:
                        iterator.close()
                    break

        # Check requirements and return accordingly
        if requires == "robust":
            # Must be robust (best_robust_logit > 0)
            if best_robust_logit < 0:
                # Not robust, return original
                return pd.DataFrame([query_instance], columns=self.data.feature_names)
        elif requires == "valid":
            # Must be valid for original model
            if best_val_term1 is None:
                # No valid CF found, return original
                return pd.DataFrame([query_instance], columns=self.data.feature_names)
            # Check if valid for target_class
            is_valid = (target_class == 1 and best_val_term1 > 0) or (target_class == 0 and best_val_term1 < 0)
            if not is_valid:
                # Not valid, return original
                return pd.DataFrame([query_instance], columns=self.data.feature_names)
        # requires == "none": return whatever we got
        
        # Return best counterfactual found
        return pd.DataFrame(best_x.cpu().numpy(), columns=self.data.feature_names)
