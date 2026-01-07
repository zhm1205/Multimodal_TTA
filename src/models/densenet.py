# densenet_simple.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import get_logger
from src.utils.config import get_config, require_config


# ---------------------- weights resolver ----------------------
def _resolve_weights(model_name: str, weights: Optional[str], pretrained: bool):
    """
    Resolve torchvision weights for both new and old APIs.
    - New API (>=0.13): pass a Weights enum instance via 'weights='
    - Old API: pass boolean 'pretrained='
    """
    enum_name = f"{model_name.replace('densenet','DenseNet')}_Weights"
    weights_enum = getattr(tvm, enum_name, None)

    # New Weights API
    if weights_enum is not None:
        if isinstance(weights, str) and hasattr(weights_enum, weights):
            return getattr(weights_enum, weights)
        if pretrained and weights is None:
            # DenseNet variants commonly use IMAGENET1K_V1 as default
            return getattr(weights_enum, "IMAGENET1K_V1")
        return None

    # Old API fallback (torchvision < 0.13)
    return pretrained  # True/False


# ---------------------- model wrapper ----------------------
class DenseNetSimple(nn.Module):
    """
    Minimal wrapper over torchvision DenseNet.

    Modes:
      - Classification (default): keep official classifier (Linear)
      - ReID: replace classifier by a projection to configurable embed_dim
              (+ optional BNNeck, optional L2-norm)

    Forward:
      - Classification: returns (features, logits)
      - ReID:           returns (features, embedding)
    """

    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        self.logger = get_logger()
        name: str = get_config(cfg, "name", "densenet121", type_=str)
        self.mode_reid: bool = bool(get_config(cfg, "reid", False))

        # num_classes needed only for classification mode
        num_classes = (
            get_config(cfg, "num_classes")
            or get_config(cfg, "task.num_classes")
            or get_config(cfg, "output_dim")
        )
        if not self.mode_reid and num_classes is None:
            raise ValueError("num_classes is required for classification "
                             "(num_classes / task.num_classes / output_dim)")

        pretrained: bool = bool(get_config(cfg, "pretrained", True))
        weights: Optional[str] = get_config(cfg, "weights", None, type_=str)
        drop_rate: float = float(get_config(cfg, "drop_rate", 0.0))  # backbone internal dropout

        # ReID-specific configs
        self._embed_dim: Optional[int] = get_config(cfg, "embed_dim", None)
        self.bnneck: bool = bool(get_config(cfg, "bnneck", False))
        self.l2_norm: bool = bool(get_config(cfg, "l2_norm", True))

        self.logger.info(
            f"Building DenseNet model: {name}, pretrained={pretrained}, "
            f"weights={weights}, drop_rate={drop_rate}, reid={self.mode_reid}, "
            f"embed_dim={self._embed_dim}, bnneck={self.bnneck}, l2_norm={self.l2_norm}"
        )

        # Build official model (pass weights= for new API, or pretrained= fallback)
        if not hasattr(tvm, name):
            raise ValueError(f"Unsupported model: {name}")
        ctor = getattr(tvm, name)
        w = _resolve_weights(name, weights, pretrained)
        try:
            self.backbone: nn.Module = ctor(weights=w, drop_rate=drop_rate)
        except TypeError:
            self.backbone = ctor(pretrained=bool(w), drop_rate=drop_rate)

        # Read pooled feature dimension from classifier
        num_ftrs = int(self.backbone.classifier.in_features)  # type: ignore[attr-defined]
        self._num_ftrs = num_ftrs
        self._num_classes = int(num_classes) if num_classes is not None else 0
        self.drop_rate = drop_rate  # we'll also allow feature-level dropout (external)

        # -------- heads --------
        if self.mode_reid:
            # ReID: remove official classifier; build projection head
            self.backbone.classifier = nn.Identity()  # type: ignore[attr-defined]

            neck_layers = []
            if self.bnneck:
                neck_layers.append(nn.BatchNorm1d(num_ftrs))
            self.neck = nn.Sequential(*neck_layers) if neck_layers else nn.Identity()

            proj_dim = int(self._embed_dim or num_ftrs)  # allow no dimension reduction
            self.proj = nn.Linear(num_ftrs, proj_dim, bias=not self.bnneck)
            nn.init.zeros_(self.proj.bias)
            self._embed_dim = proj_dim

        else:
            # Classification: single Linear head
            self.backbone.classifier = nn.Linear(num_ftrs, self._num_classes)  # type: ignore[attr-defined]
            nn.init.zeros_(self.backbone.classifier.bias)  # type: ignore[attr-defined]

        # Consistent: zero bias for all Linear layers
        for m_ in self.backbone.modules():
            if isinstance(m_, nn.Linear) and m_.bias is not None:
                nn.init.zeros_(m_.bias)

    def forward(self, x: torch.Tensor):
        """
        Return:
          - Classification: (features, logits)
          - ReID:           (features, embedding)
        """
        # Backbone feature trunk
        features = self.backbone.features(x)  # type: ignore[attr-defined]
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        # Optional dropout on pooled features (separate from DenseNet's internal drop_rate)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)

        if self.mode_reid:
            z = self.neck(features)
            z = self.proj(z)
            if self.l2_norm:
                z = F.normalize(z, p=2, dim=1)
            return features, z
        else:
            logits = self.backbone.classifier(features)  # type: ignore[attr-defined]
            return features, logits

    @property
    def num_ftrs(self):
        return self._num_ftrs

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def embed_dim(self) -> int:
        return getattr(self, "_embed_dim", self._num_ftrs)


# -------- registry helpers --------
def _with_name(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a shallow-copied cfg with the model name set."""
    new_cfg = dict(cfg)
    new_cfg["name"] = name
    return new_cfg

def DenseNet121(cfg: Dict[str, Any]) -> nn.Module: return DenseNetSimple(_with_name(cfg, "densenet121"))
def DenseNet169(cfg: Dict[str, Any]) -> nn.Module: return DenseNetSimple(_with_name(cfg, "densenet169"))
def DenseNet201(cfg: Dict[str, Any]) -> nn.Module: return DenseNetSimple(_with_name(cfg, "densenet201"))
def DenseNet161(cfg: Dict[str, Any]) -> nn.Module: return DenseNetSimple(_with_name(cfg, "densenet161"))

__all__ = [
    "DenseNetSimple",
    "DenseNet121", "DenseNet169", "DenseNet201", "DenseNet161",
]
