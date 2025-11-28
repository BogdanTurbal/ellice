import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

class Data:
    """
    Data container for ElliCE.
    Handles feature metadata, encoding, and constraints.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None,
        outcome_name: Optional[str] = None
    ):
        self.df = dataframe
        self.target_column = target_column
        
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame columns: {self.df.columns.tolist()}")
            
        self.outcome_name = outcome_name or target_column
        
        # Infer feature types if not provided
        if continuous_features is None and categorical_features is None:
            self._infer_feature_types()
        else:
            self.continuous_features = continuous_features or []
            self.categorical_features = categorical_features or []
            
        self.feature_names = [c for c in self.df.columns if c != self.target_column]
        self.immutable_features = immutable_features or []
        
        # Pre-compute bounds for continuous features
        self.bounds = {}
        for feat in self.continuous_features:
            self.bounds[feat] = (self.df[feat].min(), self.df[feat].max())
            
    def _infer_feature_types(self):
        """Simple heuristic to infer feature types."""
        self.continuous_features = []
        self.categorical_features = []
        
        for col in self.df.columns:
            if col == self.target_column:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if it looks like a one-hot encoding or integer category
                if self.df[col].nunique() < 10 and self.df[col].dtype == int:
                     self.categorical_features.append(col)
                else:
                    self.continuous_features.append(col)
            else:
                self.categorical_features.append(col)

    def get_dev_data(self) -> pd.DataFrame:
        """Returns the dataframe without the target column."""
        return self.df[self.feature_names]

    def get_target_data(self) -> pd.Series:
        return self.df[self.target_column]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for normalization logic if needed."""
        return df

