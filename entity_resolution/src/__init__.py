"""Entity-resolution project package."""

from .data_loader import load_source_table
from .data_audit import audit_dataset
from .normalization import normalize_record, build_feature_views

__all__ = [
    "load_source_table",
    "audit_dataset",
    "normalize_record",
    "build_feature_views",
]
