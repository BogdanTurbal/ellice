from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """
    Configuration for generation parameters and defaults.
    """
    # Optimization
    learning_rate: float = 0.1
    max_iterations: int = 100
    patience: int = 50
    
    # Weights
    robustness_weight: float = 1.0
    proximity_weight: float = 0.0
    
    # Defaults for unweighted groups
    default_group_weight: float = 0.5
    
    # Gumbel
    gumbel_temperature: float = 1.0
    
    # Generation Control
    early_stopping: bool = True
    progress_bar: bool = True
    gradient_mode: str = "min-max" #legacy
    allowed_ranges_closest_grad: bool = False
    
    # Constraints (Can be populated later or passed at runtime)
    features_to_vary: object = None
    permitted_range: object = None
    one_way_change: object = None
    allowed_values: object = None
    one_hot_groups: object = None
