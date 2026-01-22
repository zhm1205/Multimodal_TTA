# src/models/unet_multimodal_midfusion.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple, Sequence, Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf

from monai.networks.blocks import Convolution, ResidualUnit, UpSample
from monai.networks.layers.factories import Act, Norm

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..registry import register_model


class SpecificEncoder(nn.Module):
    """
    Specific encoder: A sequence of ResidualUnits that matches original MonaiUNet.
    Each modality has its own instance with independent parameters.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        num_res_units: int,
        act: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        curr_in = in_channels
        # Standard UNet3D encoder structure: [Enc1, Enc2, Enc3, Enc4, Bottleneck]
        # For channels [32, 64, 128, 256, 512] and strides [2, 2, 2, 2]
        # layers will be:
        # Layer 0: 1 -> 32, stride 2
        # Layer 1: 32 -> 64, stride 2
        # Layer 2: 64 -> 128, stride 2
        # Layer 3: 128 -> 256, stride 2
        # Layer 4: 256 -> 512, stride 1 (Bottleneck)
        for i, (out_ch, s) in enumerate(zip(channels, strides + [1])):
            layer = ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=curr_in,
                out_channels=out_ch,
                strides=s,
                kernel_size=3,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
            )
            self.layers.append(layer)
            curr_in = out_ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        skip_features = []
        # Standard U-Net: We need skips BEFORE downsampling at each level
        # Level 0: 160, Level 1: 80, Level 2: 40, Level 3: 20
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # 1. Run the ResidualUnit part (without stride) to get skip
                # Actually, MONAI ResidualUnit with stride=2 combines conv and downsample.
                # To match MONAI's U-Net exactly:
                # The skip feature for Level L is the output of the Encoder at Level L.
                x = layer(x)
                skip_features.append(x)
            else:
                # Bottleneck
                x = layer(x)
        
        # Global feat for auxiliary tasks
        global_feat = torch.mean(x, dim=[2, 3, 4], keepdim=True)
        return x, global_feat, skip_features


class CompositionalLayer(nn.Module):
    """Residual Fusion Layer at bottleneck."""
    def __init__(self, in_channels: int, spatial_dims: int, norm: str, act: str):
        super().__init__()
        self.fusion_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=3,
            strides=1,
            act=act,
            norm=norm,
        )
    def forward(self, f_shared: torch.Tensor, f_specific: torch.Tensor) -> torch.Tensor:
        concat_feat = torch.cat([f_shared, f_specific], dim=1)
        residual = self.fusion_conv(concat_feat)
        return f_shared + residual


class DecoderStage(nn.Module):
    """A single stage of UNet decoder: Upsample + Concat Skip + ResidualUnit."""
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        stride: int,
        num_res_units: int,
        act: str,
        norm: str,
        dropout: float,
    ):
        super().__init__()
        self.upsample = UpSample(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=stride,
            mode="nontrainable",
        )
        self.conv = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            strides=1,
            kernel_size=3,
            subunits=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@register_model("unet_multimodal_deepfusion")
@register_model("unet_multimodal_midfusion")
class MultimodalUNetDeepFusion(nn.Module):
    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        
        logger = get_logger()
        self.num_modalities = int(get_config(cfg, "num_modalities", 4))
        num_classes = int(get_config(cfg, "num_classes", 3))
        spatial_dims = int(get_config(cfg, "spatial_dims", 3))
        channels = list(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        strides = list(get_config(cfg, "strides", [2, 2, 2, 2]))
        num_res_units = int(get_config(cfg, "num_res_units", 2))
        act = get_config(cfg, "act", "RELU")
        norm = get_config(cfg, "norm", "INSTANCE")
        dropout = float(get_config(cfg, "dropout", 0.0))
        
        domain_cfg = get_config(cfg, "domain_classifier", {})
        self.domain_enabled = bool(get_config(domain_cfg, "enabled", True))
        self.domain_loss_weight = float(get_config(domain_cfg, "loss_weight", 0.1))
        
        # 1. Encoders
        self.specific_encoders = nn.ModuleList([
            SpecificEncoder(spatial_dims, 1, channels, strides, num_res_units, act, norm, dropout)
            for _ in range(self.num_modalities)
        ])
        
        # 2. Fusion
        self.fusion_layer = CompositionalLayer(channels[-1], spatial_dims, norm, act)
        self.bottleneck_reduce = nn.Conv3d(channels[-1] * self.num_modalities, channels[-1], 1, bias=False)
        
        # 3. Decoder Stages (matching channels: [512, 256, 128, 64, 32])
        self.decoder_stages = nn.ModuleList()
        # Corrected Skip Alignment for U-Net:
        # Dec 0: in 512, out 256, skip 128 (from Enc Level 2)
        # Dec 1: in 256, out 128, skip 64  (from Enc Level 1)
        # Dec 2: in 128, out 64,  skip 32  (from Enc Level 0)
        # Dec 3: in 64,  out 32,  skip 1   (from Input)
        
        skip_channels_list = [channels[2], channels[1], channels[0], 1]
        
        for i in range(len(channels) - 1):
            idx = len(channels) - 1 - i # 4, 3, 2, 1
            self.decoder_stages.append(
                DecoderStage(
                    spatial_dims, 
                    channels[idx],      # in_channels
                    skip_channels_list[i], # skip_channels (Corrected)
                    channels[idx-1],    # out_channels
                    strides[idx-1],     # stride
                    num_res_units, act, norm, dropout
                )
            )
        
        # 4. Final Output Layer
        self.final_conv = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        
        # 5. Domain Classifier
        if self.domain_enabled:
            self.domain_classifier = nn.Linear(channels[-1], self.num_modalities)

        logger.info(f"[MultimodalUNetDeepFusion] Explicitly built with {self.num_modalities} branches and UNet3D decoder stages.")

    def forward(
        self, 
        x: torch.Tensor,
        return_domain_logits: bool = False,
        return_intermediate_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        B, C, D, H, W = x.shape
        modalities = torch.split(x, 1, dim=1)
        
        specific_feats, specific_globals, all_specific_skips = [], [], []
        for encoder, modal in zip(self.specific_encoders, modalities):
            feat, glob, skips = encoder(modal)
            specific_feats.append(feat)
            specific_globals.append(glob)
            all_specific_skips.append(skips)
        
        # Pseudo-Shared Residual Fusion
        pseudo_shared_feat = torch.stack(specific_feats, dim=1).mean(dim=1)
        fused_feats = [self.fusion_layer(pseudo_shared_feat, f) for f in specific_feats]
        x_dec = self.bottleneck_reduce(torch.cat(fused_feats, dim=1))
        
        # Average Skips across modalities
        num_skips = len(all_specific_skips[0])
        fused_skips = []
        for i in range(num_skips):
            fused_skips.append(torch.stack([m[i] for m in all_specific_skips]).mean(dim=0))
        
        # Run Decoder: 
        # fused_skips contains outputs of Layer 0, 1, 2, 3
        # Resolutions: 80, 40, 20, 10
        # DecoderStage 0 (512->256) ups 10 to 20, needs skip 20 (fused_skips[2])
        # DecoderStage 1 (256->128) ups 20 to 40, needs skip 40 (fused_skips[1])
        # DecoderStage 2 (128->64) ups 40 to 80, needs skip 80 (fused_skips[0])
        # DecoderStage 3 (64->32) ups 80 to 160, needs skip 160... 
        # Wait, if we want 4 stages, we need a skip at 160 (input).
        
        # CORRECT U-NET ALIGNMENT:
        # Dec 0: Up 10->20, Concat fused_skips[2] (20)
        # Dec 1: Up 20->40, Concat fused_skips[1] (40)
        # Dec 2: Up 40->80, Concat fused_skips[0] (80)
        # Dec 3: Up 80->160, Concat Input (160)
        
        # For the final stage, we need the mean of inputs
        input_mean = torch.stack(modalities).mean(dim=0) 
        
        # Correct skip sequence for the 4 decoder stages
        skips_for_dec = [fused_skips[2], fused_skips[1], fused_skips[0], input_mean]
        
        for i, stage in enumerate(self.decoder_stages):
            x_dec = stage(x_dec, skips_for_dec[i])
        
        logits = self.final_conv(x_dec)
        
        # Constraints logic
        if return_intermediate_features and self.domain_enabled:
            shared_globals_rep = [pseudo_shared_feat.mean(dim=[2,3,4]) for _ in range(C)]
            specific_globals_flat = [g.view(B, -1) for g in specific_globals]
            return logits, shared_globals_rep, specific_globals_flat
        
        if return_domain_logits and self.domain_enabled:
            domain_logits = self.domain_classifier(torch.cat(specific_globals, dim=0).view(B*C, -1))
            return logits, domain_logits
            
        return logits

    def get_domain_loss_weight(self) -> float:
        return self.domain_loss_weight if getattr(self, 'domain_enabled', False) else 0.0
