import torch
import numpy as np
from typing import Optional, Tuple

def safe_log1pexp(x: torch.Tensor) -> torch.Tensor:
    """Stable computation of log(1+exp(x)) that avoids overflow."""
    return torch.where(x > 20, x, torch.log1p(torch.exp(x)))

def safe_log1pexp_numpy(x: np.ndarray) -> np.ndarray:
    """Stable computation of log(1+exp(x)) for numpy arrays."""
    # For large x, log(1+exp(x)) approx x
    # For small x, exp(x) is small, log1p is accurate
    # Threshold 30 ensures exp(30) doesn't overflow standard float limits easily
    # and exp(-30) is small enough.
    
    # Ideally we want log(1 + exp(-abs(x))) + max(x, 0)
    # This handles both large positive and large negative x safely.
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

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
    
    logits = H_feats @ theta_star.cpu().detach().numpy()
    # Stable sigmoid to avoid overflow
    p = np.where(logits >= 0, 
                 1.0 / (1.0 + np.exp(-logits)), 
                 np.exp(logits) / (1.0 + np.exp(logits)))
    
    # Hessian approximation
    # W = diag(p * (1-p))
    # H = X^T W X / N + reg * I
    W = H_feats * (p * (1 - p))[:, None]
    # Ensure safe division by N (number of samples)
    N = H_feats.shape[0]
    H = (W.T @ H_feats) / N + reg_coef * I
    
    # Threshold
    y_signed = 2 * y_train - 1

    # Use stable computation for loss: log(1 + exp(-y * logits))
    # Implement safety: clip logits to avoid overflow in exp if needed, though safe_log1pexp handles it.
    # But strictly aligning with reference:
    # Reference uses np.log1p(np.exp(...)) which can overflow.
    # But user asked to "implement safety in the same way". 
    # Actually, reference EllipsoidCEBase.py has safe_log1pexp for torch but unsafe numpy for init.
    # I already added safe_log1pexp_numpy.
    loss_vals = safe_log1pexp_numpy(-y_signed * logits)
    L_star = float(np.mean(loss_vals))
    theta_threshold = L_star + eps
    
    # Ellipsoid Matrix Q = H / (2 * eps)
    # Reference: self.Q = torch.from_numpy(H / (2 * self.eps)).to(self.device, self.dtype)
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
    # Reference uses rsqrt() which is 1/sqrt(w)
    Q_inv_sqrt = (V * w.clamp(min=1e-6).rsqrt().unsqueeze(0)) @ V.T
    
    return Q, Q_inv_sqrt, L_star, theta_threshold

