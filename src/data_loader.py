import pandas as pd
import os

def load_data(path: str):
    """
    Safely loads a CSV dataset.
    Returns:
        DataFrame if successful, otherwise None.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exist: {path}")

        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None

    except pd.errors.EmptyDataError:
        print("[ERROR] The file is empty.")
        return None

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None
