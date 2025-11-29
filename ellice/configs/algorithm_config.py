from dataclasses import dataclass

@dataclass
class AlgorithmConfig:
    """
    Configuration for algorithmic stability and internal constants.
    """
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

