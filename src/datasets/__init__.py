"""Dataset package initialization and registration."""

from ..registry import register_dataset

# Import the dataset you actually use (NIfTI format)
from .brats import (
    BratsMultiSourceNiftiDataset,
    BratsMultiNiftiBuilder,
)


__all__ = [
    'BratsMultiSourceNiftiDataset',
]
