import os
import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import load_dotenv

def optimize_dataframe(df: pd.DataFrame):
    """
    Automatically optimize dataframe memory usage by:
    - downcasting integers and floats,
    - converting objects to category when beneficial,
    - keeping bools as is
    """
    int_cols = df.select_dtypes(include=['integer']).columns
    if len(int_cols) > 0:
        c_mins = df[int_cols].min()
        c_maxs = df[int_cols].max()
        astype_dict = {}
        for col in int_cols:
            c_min = c_mins[col]
            c_max = c_maxs[col]
            if c_min >= -128 and c_max <= 127:
                astype_dict[col] = np.int8
            elif c_min >= -32768 and c_max <= 32767:
                astype_dict[col] = np.int16
            elif c_min >= -2147483648 and c_max <= 2147483647:
                astype_dict[col] = np.int32
            else:
                astype_dict[col] = np.int64
        df = df.astype(astype_dict)

    float_cols = df.select_dtypes(include=['floating']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')

    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(obj_cols) > 0:
        total_count = len(df)
        uniques = df[obj_cols].nunique()
        for col in obj_cols:
            if uniques[col] / total_count < 0.5:
                df[col] = df[col].astype("category")

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
