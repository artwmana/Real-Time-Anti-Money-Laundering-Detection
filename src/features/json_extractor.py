import re
from datetime import datetime
import ast
import os
import pandas as pd

from dotenv import load_dotenv

def load_data(path):
    load_dotenv()
    data_path = os.getenv("DATA_PATH") #~/Real-Time-Anti-Money-Laundering-Detection/data
    return pd.read_csv(os.path.join(os.path.expanduser(data_path), "raw/AMLNet_August_2025.csv"))

def normalize_python_json_string(s):
    # return str for working with json
    pattern = r"datetime\.datetime\((.*?)\)"

    def repl(match):
        args = match.group(1).split(',')
        nums = [int(a.strip()) for a in args]
        dt = datetime(*nums)
        return f"'{dt.isoformat()}'"

    s = re.sub(pattern, repl, s)
    return s

def parse_row(s):
    cleaned = normalize_python_json_string(s)
    return ast.literal_eval(cleaned)

def main(path):
    df = load_data(path)

    df["metadata"] = df["metadata"].apply(parse_row)
    df_meta = df['metadata'].apply(pd.Series)

    loc = df_meta["location"].apply(pd.Series)
    merch = df_meta["merchant_info"].apply(pd.Series).fillna("Unknown")
    dev = df_meta["device_info"].apply(pd.Series)
    risk = df_meta["risk_indicators"].apply(pd.Series)

    return pd.concat([df, dev.add_prefix("device_"), loc.add_prefix("loc_"), merch.add_prefix("merch_"), risk.add_prefix("risk_"), df_meta], axis=1)

if __name__ == "__main__":
    main()