import torch
from dataclasses import dataclass

@dataclass
class AlgorithmConfig:
    """
    Configuration for algorithmic stability and internal constants.
    """
    # Device Selection
    # If "auto", selects "cuda" if available, else "cpu"
    device: str = "auto"
    
    @classmethod
    def get_device(cls) -> str:
        """Resolves the actual device to use."""
        if cls.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.device

    # Numerical stability epsilon (for division, sqrt, log)
    epsilon: float = 1e-9
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    # Gumbel Softmax
    gumbel_epsilon: float = 1e-10
    
    # Ellipsoid computation
    ellipsoid_epsilon: float = 1e-6  # for eigendecomposition stability
    rsqrt_epsilon: float = 1e-6      # for rsqrt clamping
    
    # Sigmoid threshold for safe computation
    sigmoid_threshold: float = 20.0
    
    # Sparsity metric constant (C in C * Hamming + L1 distance)
    sparsity_constant: float = 100.0

