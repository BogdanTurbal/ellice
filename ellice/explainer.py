from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np

from .models import load_model
from .data import Data
from .generators.data_supported import DataSupportedGenerator
from .generators.continuous import ContinuousGenerator
from .generators.sparse_continuous import SparseContinuousGenerator
from .configs import AlgorithmConfig

class Explainer:
    """
    Main entry point for ElliCE.
    """
    
    def __init__(
        self,
        model: Any,
        data: Data,
        backend: str = 'auto',
        device: Optional[str] = None,
        backend_model_class: Optional[Any] = None
    ):
        """
        Initialize ElliCE Explainer.
        
        Args:
            model: The trained model (PyTorch nn.Module, sklearn model, or custom)
            data: ElliCE Data object containing training data
            backend: Backend type ('auto', 'pytorch', 'sklearn', 'custom')
            device: Device to use ('auto', 'cpu', 'cuda', 'mps'). If None, uses AlgorithmConfig.get_device()
            backend_model_class: Custom ModelWrapper class (required if backend='custom')
                Must be a subclass of ModelWrapper and implement required methods.
        """
        self.model = load_model(model, backend=backend, backend_model_class=backend_model_class)
        self.data = data
        
        # Resolve device
        if device is not None:
            self.device = device
        else:
            self.device = AlgorithmConfig.get_device()
        
    def generate_counterfactuals(
        self,
        query_instances: Union[pd.DataFrame, pd.Series, np.ndarray, List],
        total_CFs: int = 1,
        method: str = 'continuous',
        features_to_vary: Union[str, List[str]] = 'all',
        permitted_range: Optional[Dict[str, List[float]]] = None,
        one_way_change: Optional[Dict[str, str]] = None,
        allowed_values: Optional[Dict[str, List[float]]] = None,
        one_hot_groups: Optional[List[List[str]]] = None,
        robustness_epsilon: float = 0.01,
        regularization_coefficient: float = 1e-4,
        sparsity: bool = False,
        search_mode: str = 'filtering',
        optimization_params: Optional[Dict[str, Any]] = None,
        target_class: int = 1,
        return_probs: bool = False,
        progress_bar: bool = True
    ) -> pd.DataFrame:
        
        # Standardize Input
        if isinstance(query_instances, (pd.Series, np.ndarray, list)):
             if isinstance(query_instances, pd.Series):
                 df = query_instances.to_frame().T
             elif isinstance(query_instances, np.ndarray):
                 if query_instances.ndim == 1:
                     query_instances = query_instances.reshape(1, -1)
                 df = pd.DataFrame(query_instances, columns=self.data.feature_names)
             else:
                 df = pd.DataFrame([query_instances], columns=self.data.feature_names)
        else:
            df = query_instances

        # Initialize Generator
        gen_kwargs = {
            'model': self.model,
            'data': self.data,
            'eps': robustness_epsilon,
            'reg_coef': regularization_coefficient,
            'device': self.device
        }
        
        # Generate
        results = []
        opt_params = optimization_params or {}
        
        # Pass progress_bar via opt_params if supported by generator
        if method == 'continuous':
            opt_params['progress_bar'] = progress_bar
        
        if method == 'continuous':
            if sparsity:
                generator = SparseContinuousGenerator(**gen_kwargs)
            else:
                generator = ContinuousGenerator(**gen_kwargs)
        elif method == 'data_supported' or method == 'discrete': # 'discrete' for backward compatibility
            generator = DataSupportedGenerator(**gen_kwargs)
            # Pass sparsity and search_mode flags to generate method via opt_params
            opt_params['sparsity'] = sparsity
            opt_params['search_mode'] = search_mode
        else:
            raise ValueError(f"Unknown method: {method}")
        
        for idx, row in df.iterrows():
            try:
                cf = generator.generate(
                    query_instance=row,
                    k=total_CFs,
                    features_to_vary=features_to_vary,
                    permitted_range=permitted_range,
                    one_way_change=one_way_change,
                    allowed_values=allowed_values,
                    one_hot_groups=one_hot_groups,
                    target_class=target_class,
                    **opt_params
                )
                # Add original index for tracking
                if not cf.empty:
                    cf['original_index'] = idx
                    
                    if return_probs:
                        # Calculate probabilities
                        # Model Prob (Class 1)
                        # Use only feature columns
                        cf_features = cf[self.data.feature_names]
                        model_probs = generator.get_model_prob(cf_features)
                        
                        # Worst Case Prob (Target Class)
                        worst_probs = generator.get_worst_case_prob(cf_features, target_class=target_class)
                        
                        cf['model_prob_class_1'] = model_probs
                        cf['worst_case_prob_target'] = worst_probs
                        
                    results.append(cf)
            except Exception as e:
                # Add more context to error
                print(f"Error generating CF for index {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue? or Raise?
                raise e
            
        if not results:
            return pd.DataFrame()
            
        return pd.concat(results, ignore_index=True)
