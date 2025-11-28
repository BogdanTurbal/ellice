import torch
import numpy as np
from typing import Optional, Tuple

def safe_log1pexp(x: torch.Tensor) -> torch.Tensor:
    """Stable computation of log(1+exp(x)) that avoids overflow."""
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))

def compute_ellipsoid(
    H_feats: np.ndarray, 
    theta_star: torch.Tensor, 
    reg_coef: float, 
    eps: float,
    y_train: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Computes the ellipsoid Q matrix and its inverse square root.
    
    Args:
        H_feats: Penultimate features of training data (numpy)
        theta_star: Optimal parameters of the last layer (torch tensor)
        reg_coef: Regularization coefficient
        eps: Epsilon for robust set
        y_train: Training labels
        device: Torch device
    
    Returns:
        Q: The Q matrix defining the ellipsoid
        Q_inv_sqrt: Inverse square root of Q
        L_star: Optimal loss
        theta_threshold: Loss threshold for Rashomon set
    """
    dtype = theta_star.dtype
    m = H_feats.shape[1]
    I = np.eye(m)
    
    logits = H_feats @ theta_star.cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    
    # Hessian approximation
    # W = diag(p * (1-p))
    # H = X^T W X / N + reg * I
    W = H_feats * (p * (1 - p))[:, None]
    H = (W.T @ H_feats) / H_feats.shape[0] + reg_coef * I
    
    # Threshold
    y_signed = 2 * y_train - 1
    loss_vals = np.log1p(np.exp(-y_signed * logits))
    L_star = float(np.mean(loss_vals))
    theta_threshold = L_star + eps
    
    # Ellipsoid Matrix Q = H / (2 * eps)
    Q_np = H / (2 * eps)
    Q = torch.from_numpy(Q_np).to(device, dtype)
    
    # Inverse Square Root
    n = Q.shape[-1]
    I_torch = torch.eye(n, dtype=dtype, device=device)
    # Add small epsilon for numerical stability
    Q_stab = Q + 1e-6 * I_torch
    
    # Eigendecomposition
    w, V = torch.linalg.eigh(Q_stab)
    # Q^{-1/2} = V * w^{-1/2} * V^T
    Q_inv_sqrt = (V * w.clamp(min=1e-6).rsqrt().unsqueeze(0)) @ V.T
    
    return Q, Q_inv_sqrt, L_star, theta_threshold

