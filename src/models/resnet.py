# resnet_simple.py
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
    - New API (>=0.13): pass a Weights enum instance to constructor via 'weights='
    - Old API: pass boolean 'pretrained='
    """
    enum_name = f"{model_name.replace('resnet','ResNet')}_Weights"
    weights_enum = getattr(tvm, enum_name, None)

    # New Weights API
    if weights_enum is not None:
        if isinstance(weights, str) and hasattr(weights_enum, weights):
            return getattr(weights_enum, weights)
        if pretrained and weights is None:
            # torchvision default: V1 for 18/34, V2 for 50/101/152
            default = "IMAGENET1K_V1" if model_name in {"resnet18", "resnet34"} else "IMAGENET1K_V2"
            return getattr(weights_enum, default)
        return None

    # Old API fallback (torchvision < 0.13)
    return pretrained  # True/False


# ---------------------- model wrapper ----------------------
class ResNet(nn.Module):
    """
    Minimal wrapper over torchvision ResNet.

    Modes:
      - Classification (default): keep official head semantics (Linear only)
      - ReID: replace fc by a projection to configurable embed_dim (+ optional BNNeck, L2-norm)

    Forward:
      - Classification: returns (features, logits)
      - ReID:           returns (features, embedding)
    """

    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        self.logger = get_logger()
        name: str = get_config(cfg, "name", "resnet50", type_=str)
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
        drop_rate: float = float(get_config(cfg, "drop_rate", 0.0))

        # ReID-specific
        self._embed_dim: Optional[int] = get_config(cfg, "embed_dim", None)
        self.bnneck: bool = bool(get_config(cfg, "bnneck", False))
        self.l2_norm: bool = bool(get_config(cfg, "l2_norm", True))

        self.logger.info(
            f"Building ResNet model: {name}, pretrained={pretrained}, "
            f"weights={weights}, drop_rate={drop_rate}, reid={self.mode_reid}, "
            f"embed_dim={self._embed_dim}, bnneck={self.bnneck}, l2_norm={self.l2_norm}"
        )

        # Build official model
        if not hasattr(tvm, name):
            raise ValueError(f"Unsupported model: {name}")
        ctor = getattr(tvm, name)
        w = _resolve_weights(name, weights, pretrained)

        try:
            self.backbone: nn.Module = ctor(weights=w)
        except TypeError:
            self.backbone = ctor(pretrained=bool(w))

        # Read pooled feature dim from original fc
        num_ftrs = int(self.backbone.fc.in_features)  # type: ignore[attr-defined]
        self._num_ftrs = num_ftrs
        self._num_classes = int(num_classes) if num_classes is not None else 0
        self.drop_rate = drop_rate

        # -------- heads --------
        if self.mode_reid:
            # ReID: remove official classifier; build projection head
            self.backbone.fc = nn.Identity()  # type: ignore[attr-defined]

            neck_layers = []
            if self.bnneck:
                neck_layers.append(nn.BatchNorm1d(num_ftrs))
            self.neck = nn.Sequential(*neck_layers) if neck_layers else nn.Identity()

            proj_dim = int(self._embed_dim or num_ftrs)  # allow no dimension reduction
            self.proj = nn.Linear(num_ftrs, proj_dim, bias=not self.bnneck)
            nn.init.zeros_(self.proj.bias)
            self._embed_dim = proj_dim

        else:
            # Classification: official semantics is a single Linear layer
            self.backbone.fc = nn.Linear(num_ftrs, self._num_classes)  # type: ignore[attr-defined]
            # Bias -> 0 (harmless, stable)
            nn.init.zeros_(self.backbone.fc.bias)  # type: ignore[attr-defined]

        # Zero bias for all Linear layers to keep consistent with other wrappers
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Return:
          - Classification: (features, logits)
          - ReID:           (features, embedding)
        """
        bb = self.backbone

        # Stem + stages (mirror torchvision forward, stop before fc)
        x = bb.conv1(x); x = bb.bn1(x); x = bb.relu(x); x = bb.maxpool(x)
        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        x = bb.layer4(x)

        # Global average pooling -> pooled features
        x = bb.avgpool(x)
        features = torch.flatten(x, 1)  # (B, num_ftrs)

        # Optional dropout on pooled features (shared for both modes)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)

        if self.mode_reid:
            z = self.neck(features)
            z = self.proj(z)
            if self.l2_norm:
                z = F.normalize(z, p=2, dim=1)
            return features, z
        else:
            logits = bb.fc(features)  # type: ignore[attr-defined]
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

def ResNet18(cfg: Dict[str, Any]) -> nn.Module:  return ResNet(_with_name(cfg, "resnet18"))
def ResNet34(cfg: Dict[str, Any]) -> nn.Module:  return ResNet(_with_name(cfg, "resnet34"))
def ResNet50(cfg: Dict[str, Any]) -> nn.Module:  return ResNet(_with_name(cfg, "resnet50"))
def ResNet101(cfg: Dict[str, Any]) -> nn.Module: return ResNet(_with_name(cfg, "resnet101"))
def ResNet152(cfg: Dict[str, Any]) -> nn.Module: return ResNet(_with_name(cfg, "resnet152"))

def get_resnet_model(
    model_name: str,
    pretrained: bool = False,
    num_classes: Optional[int] = 1000,
    weights: Optional[str] = None,
    drop_rate: float = 0.0,
    # ReID options
    reid: bool = False,
    embed_dim: Optional[int] = None,
    bnneck: bool = False,
    l2_norm: bool = True,
):
    """
    Registry-style factory.

    Args:
        model_name: 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152'
        pretrained: whether to load official pretrained weights
        num_classes: classification head size (ignored if reid=True and not provided)
        weights: torchvision weights tag, e.g. 'IMAGENET1K_V2'
        drop_rate: dropout prob on pooled features
        reid: use ReID mode (projection head to embed_dim)
        embed_dim: output embedding dimension for ReID; defaults to num_ftrs if None
        bnneck: enable BNNeck (BatchNorm1d before Linear; also sets proj.bias=False)
        l2_norm: L2-normalize embeddings in ReID
    """
    cfg = {
        "name": model_name,
        "num_classes": num_classes,
        "task": {"num_classes": num_classes},
        "pretrained": pretrained,
        "weights": weights,   # None -> use enum default
        "drop_rate": drop_rate,
        "reid": reid,
        "embed_dim": embed_dim,
        "bnneck": bnneck,
        "l2_norm": l2_norm,
    }
    table = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
    }
    if model_name not in table:
        raise ValueError(f"Unsupported model: {model_name}")
    return table[model_name](cfg)


__all__ = [
    "ResNet",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
    "get_resnet_model",
]
