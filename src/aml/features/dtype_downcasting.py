import os
import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import load_dotenv

def optimize_dataframe(df: pd.DataFrame, verbose=True):
    """
    Automatically optimize dataframe memory usage by:
    - downcasting integers and floats,
    - converting objects to category when beneficial,
    - keeping bools as is
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # ---- Integers ----
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype(np.int8)
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # ---- Floats ----
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # ---- Booleans ----
        elif col_type == bool:
            continue

        # ---- Objects (strings) ----
        elif pd.api.types.is_object_dtype(col_type):
            num_unique = df[col].nunique()
            total_count = len(df[col])

            # If more than 50% values repeat → convert to category
            if num_unique / total_count < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df

def main() -> None:
    """
    Test
    """
    load_dotenv()

    data_root = Path(os.getenv("DATA_PATH", "data")).expanduser()
    csv_path = data_root / "raw/AMLNet_August_2025.csv"
    
    nrows = 100000
    df = pd.read_csv(csv_path, nrows=nrows)
    return optimize_dataframe(df)


if __name__ == "__main__":
    main()
