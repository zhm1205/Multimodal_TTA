# utils/config.py
from typing import Any, Optional, Type, TypeVar, cast
from omegaconf import DictConfig, OmegaConf

_T = TypeVar("_T")

def require_config(cfg: DictConfig, path: str, type_: Optional[Type[_T]] = None) -> _T:
    """
    Reads a required configuration value. Raises an error if it is missing or None.
    Accepts only DictConfig and optionally performs type checking.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"`cfg` must be DictConfig, got {type(cfg).__name__}")
    value = OmegaConf.select(cfg, path)
    if value is None:
        raise ValueError(f"Required configuration missing: {path}")
    if type_ is not None and not isinstance(value, type_):
        raise TypeError(f"Config '{path}' must be {type_.__name__}, got {type(value).__name__}")
    return cast(_T, value)

def get_config(cfg: DictConfig, path: str, default: Any = None, type_: Optional[Type[_T]] = None) -> Any:
    """
    Reads an optional configuration value. Returns the default value if missing.
    Accepts only DictConfig and optionally performs type checking.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"`cfg` must be DictConfig, got {type(cfg).__name__}")
    value = OmegaConf.select(cfg, path)
    value = default if value is None else value
    if type_ is not None and value is not None and not isinstance(value, type_):
        raise TypeError(f"Config '{path}' must be {type_.__name__}, got {type(value).__name__}")
    return value