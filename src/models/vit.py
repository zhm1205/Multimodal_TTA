# vit_simple.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import get_config
from omegaconf import DictConfig, OmegaConf


def _resolve_weights(model_name: str, weights: Optional[str], pretrained: bool):
    """
    Resolve torchvision weights for both new and old APIs.
    - New API (>=0.13): pass a Weights enum instance via 'weights='
    - Old API: pass boolean 'pretrained='
    """
    # Examples: ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights
    enum_name = (
        model_name.replace("vit_b_16", "ViT_B_16")
                  .replace("vit_b_32", "ViT_B_32")
                  .replace("vit_l_16", "ViT_L_16")
                  .replace("vit_l_32", "ViT_L_32")
                  .replace("vit_h_14", "ViT_H_14")
        + "_Weights"
    )
    weights_enum = getattr(tvm, enum_name, None)

    # New Weights API
    if weights_enum is not None:
        if isinstance(weights, str) and hasattr(weights_enum, weights):
            return getattr(weights_enum, weights)
        if pretrained and weights is None:
            # Try DEFAULT first if available; fall back to IMAGENET1K_V1
            try:
                return weights_enum.DEFAULT
            except Exception:
                if hasattr(weights_enum, "IMAGENET1K_V1"):
                    return getattr(weights_enum, "IMAGENET1K_V1")
        return None

    # Old API fallback (torchvision < 0.13)
    return pretrained  # True/False


def _get_in_features_from_heads(heads: nn.Module) -> int:
    """Best-effort to fetch input dim of the classifier head."""
    if isinstance(heads, nn.Linear):
        return heads.in_features
    if isinstance(heads, nn.Sequential) and len(heads) > 0:
        for m in heads.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
    raise RuntimeError("Cannot infer in_features from ViT heads module.")


class ViTSimple(nn.Module):
    """
    Minimal wrapper over torchvision ViT.

    Modes:
      - Classification (default): keep official 'heads' (Linear)
      - ReID: replace heads by projection to configurable embed_dim (+ optional BNNeck, L2-norm)

    Forward:
      - Classification: returns (features, logits)
      - ReID:           returns (features, embedding)
    """

    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.logger = get_logger()

        name: str = get_config(cfg, "name", "vit_b_16", type_=str)
        self.mode_reid: bool = bool(get_config(cfg, "reid", False))

        # num_classes 仅在分类模式下必需
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
            f"Building ViT model: {name}, pretrained={pretrained}, "
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

        # Get in_features from classifier heads
        in_features = _get_in_features_from_heads(self.backbone.heads)  # type: ignore[attr-defined]
        self._num_ftrs = int(in_features)
        self._num_classes = int(num_classes) if num_classes is not None else 0

        # ---- replace heads according to mode ----
        if self.mode_reid:
            # ReID 模式：去掉官方 heads，新增投影头
            self.backbone.heads = nn.Identity()  # type: ignore[attr-defined]

            neck_layers = []
            if self.bnneck:
                neck_layers.append(nn.BatchNorm1d(self._num_ftrs))
            self.neck = nn.Sequential(*neck_layers) if neck_layers else nn.Identity()

            proj_dim = int(self._embed_dim or self._num_ftrs)
            self.proj = nn.Linear(self._num_ftrs, proj_dim, bias=not self.bnneck)
            nn.init.zeros_(self.proj.bias)
            self._embed_dim = proj_dim
        else:
            # 分类模式：线性分类头（官方就是 Linear）
            self.backbone.heads = nn.Linear(self._num_ftrs, self._num_classes)  # type: ignore[attr-defined]
            # 与你的旧实现保持一致：线性层 bias = 0
            nn.init.zeros_(self.backbone.heads.bias)  # type: ignore[attr-defined]

        # 与你原实现一致：drop_rate 作用在 pooled features 上（可选）
        self.drop_rate = drop_rate

        # 统一把所有 Linear 的 bias 置 0（保持你原来的风格）
        for m_ in self.backbone.modules():
            if isinstance(m_, nn.Linear) and m_.bias is not None:
                nn.init.zeros_(m_.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the CLS-pooled representation before the classifier.
        Mirrors torchvision's VisionTransformer forward logic, stopping before heads.
        """
        bb = self.backbone
        x = bb._process_input(x)                                # type: ignore[attr-defined]
        n = x.shape[0]
        cls_token = bb.class_token.expand(n, -1, -1)            # type: ignore[attr-defined]
        x = torch.cat((cls_token, x), dim=1)
        x = bb.encoder(x)                                       # type: ignore[attr-defined]
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor):
        """
        Return:
          - Classification: (features, logits)
          - ReID:           (features, embedding)
        """
        features = self._forward_features(x)

        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)

        if self.mode_reid:
            z = self.neck(features)
            z = self.proj(z)
            if self.l2_norm:
                z = F.normalize(z, p=2, dim=1)
            return features, z
        else:
            logits = self.backbone.heads(features)  # type: ignore[attr-defined]
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
    new_cfg = dict(cfg)
    new_cfg["name"] = name
    return new_cfg

def ViT_B_16(cfg: Dict[str, Any]) -> nn.Module: return ViTSimple(_with_name(cfg, "vit_b_16"))
def ViT_B_32(cfg: Dict[str, Any]) -> nn.Module: return ViTSimple(_with_name(cfg, "vit_b_32"))
def ViT_L_16(cfg: Dict[str, Any]) -> nn.Module: return ViTSimple(_with_name(cfg, "vit_l_16"))
def ViT_L_32(cfg: Dict[str, Any]) -> nn.Module: return ViTSimple(_with_name(cfg, "vit_l_32"))
def ViT_H_14(cfg: Dict[str, Any]) -> nn.Module: return ViTSimple(_with_name(cfg, "vit_h_14"))

def get_vit_model(
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
        model_name: 'vit_b_16' | 'vit_b_32' | 'vit_l_16' | 'vit_l_32' | 'vit_h_14'
        pretrained: whether to load official pretrained weights
        num_classes: classification head size (ignored if reid=True and not provided)
        weights: torchvision weights tag, e.g. 'IMAGENET1K_V1'
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
        "weights": weights,
        "drop_rate": drop_rate,
        "reid": reid,
        "embed_dim": embed_dim,
        "bnneck": bnneck,
        "l2_norm": l2_norm,
    }
    table = {
        "vit_b_16": ViT_B_16,
        "vit_b_32": ViT_B_32,
        "vit_l_16": ViT_L_16,
        "vit_l_32": ViT_L_32,
        "vit_h_14": ViT_H_14,
    }
    if model_name not in table:
        raise ValueError(f"Unsupported model: {model_name}")
    return table[model_name](cfg)


__all__ = [
    "ViTSimple",
    "ViT_B_16", "ViT_B_32", "ViT_L_16", "ViT_L_32", "ViT_H_14",
    "get_vit_model",
]
