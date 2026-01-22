# efficientnet_simple.py
from __future__ import annotations
import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from omegaconf import DictConfig, OmegaConf
from src.utils.logger import get_logger
from src.utils.config import get_config


# ---------------------- weights resolver ----------------------
def _efficientnet_weights_enum_name(model_name: str) -> str:
    """
    Build torchvision Weights enum name for EfficientNet families.
    e.g. 'efficientnet_b0'   -> 'EfficientNet_B0_Weights'
         'efficientnet_v2_s' -> 'EfficientNet_V2_S_Weights'
    """
    parts = model_name.split('_')
    if not parts or parts[0] != 'efficientnet':
        raise ValueError(f"Not an EfficientNet model name: {model_name}")

    # 'efficientnet' -> 'EfficientNet', other tokens -> upper
    tokens = ['EfficientNet']
    tokens.extend([p.upper() for p in parts[1:]])
    return '_'.join(tokens) + '_Weights'


def _resolve_weights(model_name: str, weights: Optional[str], pretrained: bool):
    """
    Resolve torchvision weights for both new and old APIs.
    - New API (>=0.13): pass a Weights enum instance via 'weights='
    - Old API: pass boolean via 'pretrained='
    """
    enum_name = _efficientnet_weights_enum_name(model_name)
    weights_enum = getattr(tvm, enum_name, None)

    # New Weights API
    if weights_enum is not None:
        if isinstance(weights, str) and hasattr(weights_enum, weights):
            return getattr(weights_enum, weights)
        if pretrained and weights is None:
            # EfficientNet families typically expose IMAGENET1K_V1
            default = "IMAGENET1K_V1"
            return getattr(weights_enum, default)
        return None

    # Old API fallback (torchvision < 0.13)
    return pretrained  # True/False


# ---------------------- model wrapper ----------------------
class EfficientNet(nn.Module):
    """
    Minimal wrapper over torchvision EfficientNet (B0~B7, V2-S/M/L).

    Modes:
      - Classification (default): keep official head semantics (Dropout -> Linear)
      - ReID: replace head with a projection to configurable embed_dim (+ optional BNNeck, L2-norm)

    Forward:
      - Classification: returns (pooled_features, logits)
      - ReID:          returns (pooled_features, embedding)
    """

    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        self.logger = get_logger()

        name: str = get_config(cfg, "name", "efficientnet_b0", type_=str)
        self.mode_reid: bool = bool(get_config(cfg, "reid", False))

        # Classification num_classes (not required in ReID mode)
        num_classes = (
            get_config(cfg, "num_classes")
            or get_config(cfg, "output_dim")
        )
        if not self.mode_reid and num_classes is None:
            raise ValueError("num_classes is required for classification "
                             "(num_classes / output_dim)")

        pretrained: bool = bool(get_config(cfg, "pretrained", True))
        weights: Optional[str] = get_config(cfg, "weights", None, type_=str)
        drop_rate: float = float(get_config(cfg, "drop_rate", 0.0))

        # ReID-specific
        self._embed_dim: Optional[int] = get_config(cfg, "embed_dim", None)
        self.bnneck: bool = bool(get_config(cfg, "bnneck", False))
        self.l2_norm: bool = bool(get_config(cfg, "l2_norm", True))

        self.logger.info(
            f"Building EfficientNet model: {name}, pretrained={pretrained}, "
            f"weights={weights}, drop_rate={drop_rate}, reid={self.mode_reid}, "
            f"embed_dim={self._embed_dim}, bnneck={self.bnneck}, l2_norm={self.l2_norm}"
        )

        if not hasattr(tvm, name):
            raise ValueError(f"Unsupported model: {name}")
        ctor = getattr(tvm, name)
        w = _resolve_weights(name, weights, pretrained)

        # Try new API first (weights=), fall back to old (pretrained=)
        try:
            self.backbone: nn.Module = ctor(weights=w)
        except TypeError:
            self.backbone = ctor(pretrained=bool(w))

        # Figure out pooled feature dim from official classifier's last Linear
        classifier = getattr(self.backbone, "classifier", None)
        if classifier is None:
            raise RuntimeError(f"{name} has no attribute 'classifier' (unexpected torchvision version).")

        last_linear = classifier[-1] if isinstance(classifier, nn.Sequential) else classifier
        if not isinstance(last_linear, nn.Linear):
            raise RuntimeError(f"Unexpected classifier tail: {type(last_linear)}; expected nn.Linear.")

        num_ftrs = int(last_linear.in_features)
        self._num_ftrs = num_ftrs
        self._num_classes = int(num_classes) if num_classes is not None else 0
        self.drop_rate = drop_rate

        # -------- heads --------
        if self.mode_reid:
            # ReID: remove official classifier; build projection head
            self.backbone.classifier = nn.Identity()

            neck_layers = []
            if self.bnneck:
                neck_layers.append(nn.BatchNorm1d(num_ftrs))
            self.neck = nn.Sequential(*neck_layers) if neck_layers else nn.Identity()

            proj_dim = int(self._embed_dim or num_ftrs)  # allow no dimension reduction
            self.proj = nn.Linear(num_ftrs, proj_dim, bias=not self.bnneck)
            # Bias to 0; weight init (xavier/uniform) can vary—keep simple & stable
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
            self._embed_dim = proj_dim

        else:
            # Classification: keep official semantics => [Dropout(p), Linear]
            head = nn.Sequential(
                nn.Dropout(p=self.drop_rate, inplace=True),
                nn.Linear(num_ftrs, self._num_classes),
            )
            # Official-style init for Linear: U(-1/sqrt(out), +1/sqrt(out)), bias=0
            lin: nn.Linear = head[-1]
            bound = 1.0 / math.sqrt(lin.out_features)
            nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.zeros_(lin.bias)

            self.backbone.classifier = head

    # -------- forward --------
    def forward(self, x: torch.Tensor):
        """
        Returns:
          - Classification: (features, logits)
          - ReID:           (features, embedding)
        """
        bb = self.backbone
        x = bb.features(x)
        if hasattr(bb, "avgpool") and bb.avgpool is not None:
            x = bb.avgpool(x)
        else:
            x = F.adaptive_avg_pool2d(x, 1)
        features = torch.flatten(x, 1)

        if self.mode_reid:
            # We don't use official classifier, so apply dropout here to mimic its regularization role.
            if self.drop_rate > 0:
                features = F.dropout(features, p=self.drop_rate, inplace=False, training=self.training)

            z = self.neck(features)
            z = self.proj(z)
            if self.l2_norm:
                z = F.normalize(z, p=2, dim=1)
            return features, z

        else:
            logits = bb.classifier(features)
            return features, logits

    # -------- properties --------
    @property
    def num_ftrs(self) -> int:
        return self._num_ftrs

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def embed_dim(self) -> int:
        return getattr(self, "_embed_dim", self._num_ftrs)


# ---------------------- registry helpers ----------------------
def _with_name(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    new_cfg = dict(cfg)
    new_cfg["name"] = name
    return new_cfg


# EfficientNet-B0~B7
def EfficientNetB0(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b0"))
def EfficientNetB1(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b1"))
def EfficientNetB2(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b2"))
def EfficientNetB3(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b3"))
def EfficientNetB4(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b4"))
def EfficientNetB5(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b5"))
def EfficientNetB6(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b6"))
def EfficientNetB7(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_b7"))

# EfficientNet-V2 S/M/L
def EfficientNetV2S(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_v2_s"))
def EfficientNetV2M(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_v2_m"))
def EfficientNetV2L(cfg: Dict[str, Any]) -> nn.Module: return EfficientNet(_with_name(cfg, "efficientnet_v2_l"))


def get_efficientnet_model(
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
        model_name: 'efficientnet_b0'~'efficientnet_b7' | 'efficientnet_v2_s/m/l'
        pretrained: whether to load official pretrained weights
        num_classes: classification head size (ignored if reid=True and not provided)
        weights: torchvision weights tag, e.g. 'IMAGENET1K_V1'
        drop_rate: dropout prob (classification head, or features-dropout in ReID)
        reid: use ReID mode (projection head to embed_dim)
        embed_dim: output embedding dimension for ReID; defaults to num_ftrs if None
        bnneck: enable BNNeck (BatchNorm1d before Linear; also sets proj.bias=False)
        l2_norm: L2-normalize embeddings in ReID
    """
    cfg = {
        "name": model_name,
        "num_classes": num_classes,
        "pretrained": pretrained,
        "weights": weights,
        "drop_rate": drop_rate,
        "reid": reid,
        "embed_dim": embed_dim,
        "bnneck": bnneck,
        "l2_norm": l2_norm,
    }

    table = {
        "efficientnet_b0": EfficientNetB0,
        "efficientnet_b1": EfficientNetB1,
        "efficientnet_b2": EfficientNetB2,
        "efficientnet_b3": EfficientNetB3,
        "efficientnet_b4": EfficientNetB4,
        "efficientnet_b5": EfficientNetB5,
        "efficientnet_b6": EfficientNetB6,
        "efficientnet_b7": EfficientNetB7,
        "efficientnet_v2_s": EfficientNetV2S,
        "efficientnet_v2_m": EfficientNetV2M,
        "efficientnet_v2_l": EfficientNetV2L,
    }
    if model_name not in table:
        raise ValueError(f"Unsupported model: {model_name}")
    return table[model_name](cfg)


__all__ = [
    "EfficientNet",
    "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
    "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
    "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L",
    "get_efficientnet_model",
]
