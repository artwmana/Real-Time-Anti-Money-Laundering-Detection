from .dtype_downcasting import optimize_dataframe
from .json_extractor import flatten_metadata
from .feature_pipeline import FeaturePipeline
from .encoding import SmoothedTargetEncoder

__all__ = [
    "optimize_dataframe", 
    "flatten_metadata",
    "FeaturePipeline",
    "SmoothedTargetEncoder",
]
