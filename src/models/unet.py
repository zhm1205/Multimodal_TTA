# src/models/generators/unet_monai_delta.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from monai.networks.nets import UNet as MonaiUNet

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.registry import register_model


@register_model("unet")
class UNet(MonaiUNet):
    def __init__(
        self,
        cfg: DictConfig | Dict[str, Any],
        in_channels: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        log = get_logger()

        # 读配置
        c_in_cfg = get_config(cfg, "in_channels", 3)
        c_in = (
            in_channels
            if in_channels is not None
            else (None if c_in_cfg == "auto" else int(c_in_cfg))
        )
        if c_in is None:
            raise ValueError(
                "[UNet] in_channels is 'auto'; please pass in_channels at construction time."
            )

        out_ch = int(get_config(cfg, "num_classes", 1))

        channels = list(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        strides = list(get_config(cfg, "strides", [2, 2, 2, 2]))
        num_res_units = int(get_config(cfg, "num_res_units", 0))
        act = get_config(cfg, "act", "relu")
        norm = get_config(cfg, "norm", "BATCH")
        dropout = float(get_config(cfg, "dropout", 0.0))

        # 新增：从 cfg 里读 spatial_dims，默认 3（3D 分割）
        spatial_dims = int(get_config(cfg, "spatial_dims", 3))

        log.info(
            f"[Gen] UNet: spatial_dims={spatial_dims}, in={c_in}, out={out_ch}, "
            f"channels={channels}, strides={strides}, res_units={num_res_units}, "
            f"act={act}, norm={norm}, dropout={dropout}"
        )

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=c_in,
            out_channels=out_ch,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
