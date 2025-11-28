import yaml
import json
from typing import Dict, Any, Tuple, List
from pathlib import Path

def load_config(path: str, dataset_name: str) -> Tuple[List[str], Dict[str, List[float]], Dict[str, str]]:
    """
    Load actionability configuration from a YAML or JSON file.
    
    Args:
        path: Path to the config file.
        dataset_name: Key for the dataset in the config.
        
    Returns:
        (immutable_features, permitted_range, one_way_change)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(p, 'r') as f:
        if p.suffix in ['.yml', '.yaml']:
            config = yaml.safe_load(f)
        elif p.suffix == '.json':
            config = json.load(f)
        else:
            # Try yaml default
            config = yaml.safe_load(f)
            
    if dataset_name not in config:
        # Maybe the file IS the config for the dataset?
        # If not found, warn or error.
        # Let's assume strictly nested structure as in demo.
        print(f"Warning: {dataset_name} not found in config. Trying root.")
        entry = config
    else:
        entry = config[dataset_name]
        
    immutables = entry.get('immutables', [])
    ranges = entry.get('acceptable_ranges', {})
    one_way = entry.get('one_way_change', {})
    
    return immutables, ranges, one_way
