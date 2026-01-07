"""Evaluation strategy package."""

# Import strategies so that they register themselves with the global registry
from . import brats_eval  # noqa: F401


__all__ = [
    'brats_eval',
]
