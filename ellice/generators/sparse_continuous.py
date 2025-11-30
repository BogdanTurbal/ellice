import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from tqdm import tqdm
import torch.optim as optim

from ..configs import AlgorithmConfig, GenerationConfig
from .continuous import ContinuousGenerator

class SparseContinuousGenerator(ContinuousGenerator):
    """
    Sparse Continuous Counterfactual Generator.
    
    Implements Algorithm 3 from the ElliCE paper:
    1. Runs full optimization to determine gradient magnitudes.
    2. Ranks features by importance.
    3. Iteratively adds features (greedy approach) and optimizes with masked gradients.
    4. Stops as soon as a robust counterfactual is found with minimum active features.
    """
    
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
        
        # --- Step 1: Full Optimization & Gradient Accumulation ---
        
        # Setup Optimization Variables
        setup = self._initialize_optimization_setup(
            query_instance, feature_weights, one_hot_groups, learning_rate, features_to_vary
        )
        
        x_orig = setup['x_orig']
        input_dim = setup['input_dim']
        loss_weights = setup['loss_weights']
        one_hot_indices = setup['one_hot_indices']
        continuous_indices = setup['continuous_indices']
        x_cont = setup['x_cont']
        cat_logits_list = setup['cat_logits_list']
        params = setup['params']
        optimizer = setup['optimizer']
        frozen_indices_cont = setup['frozen_indices_cont']
        frozen_groups_cat = setup['frozen_groups_cat']
        
        # Accumulator for gradients
        grad_accumulator = torch.zeros(input_dim, device=self.device, dtype=self.dtype)
        
        # Setup Constraints
        range_constraints_cont, allowed_values_cont, one_way_constraints_cont = self._initialize_constraints(
            continuous_indices, permitted_range, allowed_values, one_way_change
        )
        
        # Pre-compute constant
        inv_sqrt = self.Q_inv_sqrt
        bias_c = torch.ones(1, 1, device=self.device, dtype=self.dtype)

        # --- Optimization Loop for Gradient Accumulation ---
        # We iterate for 'max_iterations' or until convergence, accumulating grads
        
        for i in range(max_iterations):
            optimizer.zero_grad()
            
            # Construct x_cf
            x_cf_full = torch.zeros_like(x_orig)
            if x_cont is not None:
                x_cf_full[0, continuous_indices] = x_cont
            if cat_logits_list:
                for idx, group_indices in enumerate(one_hot_indices):
                    if idx in frozen_groups_cat:
                        orig_vals = x_orig[0, group_indices]
                        x_cf_full[0, group_indices] = orig_vals
                    else:
                        probs = self._gumbel_softmax_sample(cat_logits_list[idx], tau=gumbel_temperature)
                        x_cf_full[0, group_indices] = probs
            
            # Forward & Loss
            h_flat = self._get_penult_features(x_cf_full)
            h_aug = torch.cat([h_flat, bias_c], dim=1)
            
            with torch.no_grad():
                worst_theta = self._compute_worst_model(h_aug)
                
            if target_class == 0:
                # Adjust worst_theta logic for class 0 target
                u = inv_sqrt @ h_aug.T 
                norm_u = u.norm()
                direction = inv_sqrt @ u / (norm_u + AlgorithmConfig.epsilon)
                worst_theta = self.omega_c + direction.squeeze()

            robust_logit = torch.matmul(h_aug, worst_theta)
            
            if target_class == 1:
                loss_robust = -robust_logit
            else:
                loss_robust = robust_logit
                
            diff = x_cf_full - x_orig
            weighted_diff_sq = loss_weights * (diff ** 2)
            loss_prox = torch.sqrt(torch.sum(weighted_diff_sq) + AlgorithmConfig.epsilon)
            
            loss = robustness_weight * loss_robust + proximity_weight * loss_prox
            loss.backward()
            
            # Accumulate Gradients
            # Continuous
            if x_cont is not None and x_cont.grad is not None:
                grad_accumulator[continuous_indices] += x_cont.grad.abs()
            
            # Categorical
            if cat_logits_list:
                for idx, logits in enumerate(cat_logits_list):
                    if logits.grad is not None:
                        # Assign max or sum grad to the group features?
                        # Usually sparsity is defined on the original input space.
                        # We'll distribute the mean grad to the group indices for ranking
                        g = logits.grad.abs().mean()
                        grad_accumulator[one_hot_indices[idx]] += g

            # Clip & Step
            torch.nn.utils.clip_grad_norm_(params, max_norm=AlgorithmConfig.clip_grad_norm)
            optimizer.step()
            
            # 4. Projections / Constraints (Only on Continuous Variables)
            self._project_continuous_features(
                x_cont, x_orig, continuous_indices, frozen_indices_cont,
                allowed_values_cont, range_constraints_cont, one_way_constraints_cont
            )
            

        # --- Step 2: Rank Features ---
        # Sort indices by accumulated gradient magnitude (descending)
        sorted_indices = torch.argsort(grad_accumulator, descending=True).cpu().numpy()
        
        # Filter out immutable features from sorted_indices
        frozen_global_indices = set()
        for local_idx in frozen_indices_cont:
            frozen_global_indices.add(continuous_indices[local_idx])
        
        for group_idx in frozen_groups_cat:
            for global_idx in one_hot_indices[group_idx]:
                frozen_global_indices.add(global_idx)
                
        sorted_indices = [idx for idx in sorted_indices if idx not in frozen_global_indices]
        
        # --- Step 3: Iterative Selection (Greedy) ---
        # active_set = set()
        # Reset solution
        
        # We iterate adding features one by one (or group by group)
        
        active_mask = torch.zeros(input_dim, device=self.device, dtype=torch.bool)
        
        # We need a new optimization loop for EACH K
        # This can be expensive. We optimize until robustness is achieved.
        
        # Filter indices: remove immutable ones first? 
        # Or just iterate all. Immutable ones should have 0 gradient if logic was correct (but gradients flow through model).
        # We should explicitly ignore immutable features in 'active_mask' logic if they are frozen.
        
        # Let's iterate through sorted features
        final_cf = None
        best_valid_cf = None
        
        if progress_bar:
            print("Running Sparse Optimization...")
            
        # We try K = 1 to N features
        for k in range(1, len(sorted_indices) + 1):
            #print(f"Trying {k} features...")
            idx_to_add = sorted_indices[k-1]
            active_mask[idx_to_add] = True
            
            # If this index belongs to a group, enable the whole group?
            if one_hot_groups:
                for grp in one_hot_indices:
                    if idx_to_add in grp:
                        active_mask[grp] = True
                        break
            
            
            params_k = []
            
            # Continuous subset
            active_cont_indices = [i for i in continuous_indices if active_mask[i]]
            if active_cont_indices:
                # We optimize a smaller tensor
                x_cont_k = x_orig[0, active_cont_indices].clone().detach().requires_grad_(True)
                params_k.append(x_cont_k)
            else:
                x_cont_k = None
                
            # Categorical subset
            # Only include logits for groups that are active
            active_cat_groups = []
            cat_logits_k = []
            if one_hot_indices:
                for grp_idx, grp in enumerate(one_hot_indices):
                    # If any feature in group is active, the group is active (due to logic above)
                    if active_mask[grp[0]]: # Check first index
                        current_vals = x_orig[0, grp]
                        active_idx = torch.argmax(current_vals).item()
                        logits = torch.full((len(grp),), 0.1, device=self.device, dtype=self.dtype, requires_grad=True)
                        logits.data[active_idx] = 1.0
                        cat_logits_k.append(logits)
                        active_cat_groups.append(grp_idx)
                        params_k.append(logits)
            
            if not params_k:
                continue # No active mutable features yet
                
            optimizer_k = optim.Adam(params_k, lr=learning_rate)
            
            # Inner Optimization Loop
            found_robust = False
            
            # We run for fewer iterations or until robust
            # Paper suggests "masked optimization".
            
            for step in range(max_iterations):
                optimizer_k.zero_grad()
                
                # Reconstruct full x_cf
                x_cf_curr = x_orig.clone() # Start with original
                
                # Update continuous
                if x_cont_k is not None:
                    x_cf_curr[0, active_cont_indices] = x_cont_k
                    
                # Update categorical
                if cat_logits_k:
                    for lg_idx, logits in enumerate(cat_logits_k):
                        grp_indices = one_hot_indices[active_cat_groups[lg_idx]]
                        probs = self._gumbel_softmax_sample(logits, tau=gumbel_temperature)
                        x_cf_curr[0, grp_indices] = probs
                        
                # Loss Calculation (Standard)
                h_flat = self._get_penult_features(x_cf_curr)
                h_aug = torch.cat([h_flat, bias_c], dim=1)
                
                # Check original model logit for "valid" requirement
                with torch.no_grad():
                    val_term1 = torch.matmul(h_aug, self.omega_c).item()
                    worst_theta = self._compute_worst_model(h_aug)
                
                if target_class == 0:
                    u = inv_sqrt @ h_aug.T 
                    norm_u = u.norm()
                    direction = inv_sqrt @ u / (norm_u + AlgorithmConfig.epsilon)
                    worst_theta = self.omega_c + direction.squeeze()

                robust_logit = torch.matmul(h_aug, worst_theta)
                robust_logit_val = robust_logit.item()
                
                if target_class == 1:
                    loss_robust = F.relu(-robust_logit)
                    is_robust = robust_logit_val > 0
                    is_valid_orig = val_term1 > 0
                else:
                    loss_robust = F.relu(robust_logit)
                    is_robust = robust_logit_val < 0
                    is_valid_orig = val_term1 < 0
                
                # Check if meets requirements
                # 1. Robustness (Stop immediately)
                if is_robust:
                    found_robust = True
                    final_cf = x_cf_curr.detach()
                    break
                
                # 2. Validity (Store best/first found, continue searching for robust)
                if is_valid_orig:
                    if best_valid_cf is None:
                        best_valid_cf = x_cf_curr.detach()
                
                diff = x_cf_curr - x_orig
                weighted_diff_sq = loss_weights * (diff ** 2)
                loss_prox = torch.sqrt(torch.sum(weighted_diff_sq) + AlgorithmConfig.epsilon)
                
                loss = robustness_weight * loss_robust + proximity_weight * loss_prox
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(params_k, max_norm=AlgorithmConfig.clip_grad_norm)
                optimizer_k.step()
                
                # Projections / Constraints for sparse loop
                with torch.no_grad():
                    if x_cont_k is not None:
                        # Global bounds check first (clipping)
                        for k_idx, global_idx in enumerate(active_cont_indices):
                             min_val = self.feature_mins[global_idx]
                             max_val = self.feature_maxs[global_idx]
                             x_cont_k.data[k_idx] = torch.clamp(x_cont_k.data[k_idx], min_val, max_val)

                        # 1. Enforce min/max of allowed values
                        for idx, vals in allowed_values_cont:
                            global_idx = continuous_indices[idx]
                            if global_idx in active_cont_indices:
                                k_idx = active_cont_indices.index(global_idx)
                                min_v, max_v = vals[0], vals[-1]
                                x_cont_k.data[k_idx] = torch.clamp(x_cont_k.data[k_idx], min_v, max_v)
                        
                        # Clip to permitted ranges
                        for idx, min_v, max_v in range_constraints_cont:
                            global_idx = continuous_indices[idx]
                            if global_idx in active_cont_indices:
                                k_idx = active_cont_indices.index(global_idx)
                                x_cont_k.data[k_idx] = torch.clamp(x_cont_k.data[k_idx], min_v, max_v)
                        
                        # Enforce one-way changes
                        for idx, direction in one_way_constraints_cont:
                            global_idx = continuous_indices[idx]
                            if global_idx in active_cont_indices:
                                k_idx = active_cont_indices.index(global_idx)
                                orig_val = x_orig.data[0, global_idx]
                                if direction == 'increase':
                                     x_cont_k.data[k_idx] = torch.max(x_cont_k.data[k_idx], orig_val)
                                elif direction == 'decrease':
                                     x_cont_k.data[k_idx] = torch.min(x_cont_k.data[k_idx], orig_val)
                        
                        # Snap to allowed values (Hard constraint)
                        for idx, vals in allowed_values_cont:
                            global_idx = continuous_indices[idx]
                            if global_idx in active_cont_indices:
                                k_idx = active_cont_indices.index(global_idx)
                                val = x_cont_k.data[k_idx]
                                diff = (vals - val).abs()
                                min_idx = torch.argmin(diff)
                                x_cont_k.data[k_idx] = vals[min_idx]
            
            if found_robust:
                if progress_bar:
                    req_text = "robust" if requires == "robust" else "valid" if requires == "valid" else "CF"
                    print(f"{req_text.capitalize()} CF found with {k} active features (or groups).")
                break
        
        if final_cf is None:
            # No robust CF found. Check if we found a valid one and if that satisfies requirements.
            if (requires == "valid" or requires == "none") and best_valid_cf is not None:
                final_cf = best_valid_cf
            
            # If still None, we failed
            elif requires in ["valid", "robust"]:
                import warnings
                warnings.warn(
                    f"\n Failed to find {requires} counterfactual after trying all features. "
                    f"Try to increase regularization_coefficient or decrease robustness_epsilon",
                    UserWarning
                )
                # Return original query instance
                return pd.DataFrame([query_instance], columns=self.data.feature_names)
            else:
                # requires == "none": return empty
                return pd.DataFrame()
        
        # Verify final_cf meets requirements (double-check)
        if requires in ["valid", "robust"]:
            # Check if final_cf is actually valid/robust
            final_cf_df = pd.DataFrame(final_cf.cpu().numpy(), columns=self.data.feature_names)
            final_cf_tensor = torch.tensor(final_cf_df.values, dtype=self.dtype, device=self.device)
            
            with torch.no_grad():
                h_flat = self._get_penult_features(final_cf_tensor)
                bias = torch.ones(h_flat.size(0), 1, device=self.device, dtype=self.dtype)
                h_aug = torch.cat([h_flat, bias], dim=1)
                
                val_term1 = torch.matmul(h_aug, self.omega_c).item()
                worst_theta = self._compute_worst_model(h_aug)
                robust_logit = torch.matmul(h_aug, worst_theta).item()
                
                if requires == "robust":
                    is_robust = (target_class == 1 and robust_logit > 0) or (target_class == 0 and robust_logit < 0)
                    if not is_robust:
                        return pd.DataFrame([query_instance], columns=self.data.feature_names)
                elif requires == "valid":
                    is_valid = (target_class == 1 and val_term1 > 0) or (target_class == 0 and val_term1 < 0)
                    if not is_valid:
                        return pd.DataFrame([query_instance], columns=self.data.feature_names)
                    
                    # Check if valid but not robust - warn user
                    is_robust = (target_class == 1 and robust_logit > 0) or (target_class == 0 and robust_logit < 0)
                    if not is_robust:
                        import warnings
                        warnings.warn(
                            f"\n Found valid counterfactual but it is not robust (robust_logit={robust_logit:.4f}). "
                            f"Try to increase regularization_coefficient or decrease robustness_epsilon for better robustness.",
                            UserWarning
                        )
            
        return pd.DataFrame(final_cf.cpu().numpy(), columns=self.data.feature_names)

