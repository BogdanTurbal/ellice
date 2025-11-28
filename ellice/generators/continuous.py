import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
import time

from tqdm import tqdm


from .base import EllipsoidGenerator

class ContinuousGenerator(EllipsoidGenerator):
    """
    Gradient-based Counterfactual Generator.
    Optimizes the input features directly to find a robust counterfactual.
    """
    
    def _gumbel_softmax_sample(self, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Sample from Gumbel-Softmax distribution."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)

    def generate(
        self,
        query_instance: pd.Series,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        robustness_weight: float = 1.0,
        proximity_weight: float = 0.0,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[float]]] = None,
        one_way_change: Optional[Dict[str, str]] = None,
        one_hot_groups: Optional[List[List[str]]] = None,
        gumbel_temperature: float = 1.0,
        target_class: int = 1,
        early_stopping: bool = True,
        patience: int = 20,
        progress_bar: bool = True,
        gradient_mode: str = "min-max", # "min-max" or "full_grad"
        **kwargs
    ) -> pd.DataFrame:
        
        # Prepare Input
        x_orig = torch.tensor(query_instance.values, dtype=self.dtype, device=self.device).unsqueeze(0)
        feature_names = self.data.feature_names
        input_dim = len(feature_names)
        
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
                
                logits = torch.zeros(len(group_indices), device=self.device, dtype=self.dtype, requires_grad=True)
                logits.data[active_idx] = 2.0 # Strong bias towards original
                
                cat_logits_list.append(logits)
                params.append(logits)
        
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Target
        target = torch.tensor([float(target_class)], device=self.device, dtype=self.dtype)
        
        # Best tracking
        best_x = x_orig.clone().detach()
        best_robust_logit = -float('inf')
        
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
        
        # Pre-compute indices for constraints to speed up loop
        range_constraints_cont = [] # (local_cont_idx, min, max)
        if permitted_range:
            for feat, (min_v, max_v) in permitted_range.items():
                if feat in feature_names:
                    global_idx = feature_names.index(feat)
                    if global_idx in continuous_indices:
                        local_idx = continuous_indices.index(global_idx)
                        range_constraints_cont.append((local_idx, min_v, max_v))
                    
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
        if progress_bar:
            print("Progress Bar Enabled")
            iterator = tqdm(iterator, desc="Generating CF", leave=True, mininterval=0)

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
            
            # 2. Compute Robust Logit
            if gradient_mode == "full_grad":
                u = inv_sqrt @ h_aug.T
                term2 = torch.norm(u, p=2)
                term1 = torch.matmul(h_aug, self.omega_c)
                
                robust_logit = term1 - term2
                if target_class == 0:
                    robust_logit = term1 + term2
                    
            else: # "min-max" default
                with torch.no_grad():
                    worst_theta = self._compute_worst_model(h_aug)
                    
                if target_class == 0:
                    u = inv_sqrt @ h_aug.T 
                    norm_u = u.norm()
                    direction = inv_sqrt @ u / (norm_u + 1e-9)
                    worst_theta = self.omega_c + direction.squeeze()
                    
                robust_logit = torch.matmul(h_aug, worst_theta)
            
            # Loss Logic
            if target_class == 1:
                loss_robust = F.relu( -robust_logit) 
            else:
                loss_robust = F.relu( robust_logit)

            # 3. Proximity Loss
            loss_prox = torch.norm(x_cf_full - x_orig, p=2)
            
            # Total Loss
            loss = robustness_weight * loss_robust + proximity_weight * loss_prox
            
            loss.backward()
            optimizer.step()
            
            # 4. Projections / Constraints (Only on Continuous Variables)
            with torch.no_grad():
                if x_cont is not None:
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

                # Update Progress Bar
                if progress_bar:
                    iterator.set_postfix({
                        'Prob': f"{current_prob:.3f}",
                        'RobProb': f"{robust_prob:.3f}",
                        'BestRobLogit': f"{best_robust_logit:.3f}"
                    })

                # Save best
                if metric > best_robust_logit:
                    best_robust_logit = metric
                    best_x = x_val_full.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Early Stopping
                if early_stopping and no_improve_count >= patience:
                    if progress_bar:
                        iterator.close()
                    break

        return pd.DataFrame(best_x.cpu().numpy(), columns=self.data.feature_names)
