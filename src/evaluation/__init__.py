"""Evaluation strategy package."""

# Import strategies so that they register themselves with the global registry
from . import seg_eval  # noqa: F401


__all__ = [
    'seg_eval',
]
