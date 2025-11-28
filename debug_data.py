import pandas as pd
import numpy as np
from ellice.ellice.data import Data

def debug_data():
    print("Debugging Data class...")
    # Create dummy data
    X = np.random.rand(5, 3)
    df = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
    df['target'] = [0, 1, 0, 1, 0]
    
    print(f"Columns: {df.columns.tolist()}")
    
    data = Data(df, target_column='target')
    print(f"Target Column: {data.target_column}")
    print(f"Feature Names: {data.feature_names}")
    print(f"Feature Names Count: {len(data.feature_names)}")
    
    dev_data = data.get_dev_data()
    print(f"Dev Data Shape: {dev_data.shape}")
    
    if dev_data.shape[1] != 3:
        print("ERROR: Dev data has wrong number of columns!")
    else:
        print("SUCCESS: Dev data has correct columns.")

if __name__ == "__main__":
    debug_data()

