"""Dataset package initialization and registration."""

from ..registry import register_dataset

# Import the dataset you actually use (NIfTI format)
from .brats import (
    BratsMultiSourceNiftiDataset as BratsDataset,
    BratsMultiNiftiBuilder as BratsBuilder,
)

from .hecktor21 import (
    Hecktor21Dataset,
    Hecktor21Builder,
)


__all__ = [
    'BratsDataset',
    'Hecktor21Dataset',
    'Hecktor21Builder',
    'BratsBuilder',
]
