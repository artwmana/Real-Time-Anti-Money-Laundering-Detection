# main.py
import logging
from pathlib import Path
import pandas as pd

from aml.features.feature_pipeline import FeaturePipeline


# 1. Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,   # DEBUG если нужно больше подробностей
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "feature_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# 2. Main entry
def main():
    logger.info("Application started")

    df = pd.read_csv("data/train.csv")
    y = df["isFraud"]

    fp = FeaturePipeline(
        enable_downcasting=True,
        enable_encoding=True,
        enable_scale=True,
    )

    X = fp.fit_transform(df, y)

    logger.info("Pipeline finished | shape=%s", X.shape)


if __name__ == "__main__":
    main()
