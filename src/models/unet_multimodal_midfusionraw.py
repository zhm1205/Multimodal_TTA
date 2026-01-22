# src/models/unet_multimodal_midfusion.py
"""
Multi-modal UNet with Shared-Specific Deep Feature Fusion

Architecture based on DualNet_SS fusion strategy with UNet3D encoders:
    - Shared Encoder: Full UNet3D encoder (5 layers) for cross-modality features
    - Specific Encoders: 4 independent full UNet3D encoders for modality-specific features
    - Residual Fusion: fused = shared + Conv(concat(shared, specific)) at bottleneck
    - Domain Classifier: auxiliary task using specific encoder bottleneck features (optional)

Input: [B, 4, D, H, W] (4 MRI modalities: t1n, t1c, t2w, t2f)
Output: [B, 3, D, H, W] (ET/TC/WT segmentation)

Key Features:
    1. Dual-encoder architecture (1 shared + 4 specific full UNet3D encoders)
    2. Deep feature fusion at bottleneck level [D/16, H/16, W/16]
    3. Residual fusion for combining shared and specific features
    4. Domain classifier for modality discrimination (can be disabled via domain_loss_weight=0)
    5. Compatible with existing MMDG training pipeline
"""
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


class SharedEncoder(nn.Module):
    """
    Shared encoder: Full UNet3D encoder (5 layers)
    
    Processes all modalities through a single full encoder to learn cross-modality features.
    Input is reshaped from [B, 4, D, H, W] to [B*4, 1, D, H, W] for batch processing.
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 1),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "RELU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        in_ch = 1  # Single modality input
        for out_ch, stride in zip(channels, strides):
            layer = ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=stride,
                kernel_size=3,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
            )
            self.layers.append(layer)
            in_ch = out_ch
        
        self.out_channels = out_ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B*4, 1, D, H, W] - All modalities stacked in batch dimension
        Returns:
            output: [B*4, 512, D/16, H/16, W/16] - Shared bottleneck features
            global_feat: [B*4, 512, 1, 1, 1] - Global pooled features for domain classifier
            skip_features: List of intermediate features for skip connections
        """
        skip_features = []
        for layer in self.layers:
            x = layer(x)
            skip_features.append(x)
        
        # Global average pooling for domain classifier
        global_feat = torch.mean(x, dim=[2, 3, 4], keepdim=True)  # [B*4, 512, 1, 1, 1]
        
        return x, global_feat, skip_features


class SpecificEncoder(nn.Module):
    """
    Specific encoder: Full UNet3D encoder (5 layers) for single modality
    
    Each modality has its own independent full encoder to capture modality-specific characteristics.
    """
    
    def __init__(
        self,
        spatial_dims: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 1),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "RELU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        in_ch = 1  # Single modality input
        for out_ch, stride in zip(channels, strides):
            layer = ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=stride,
                kernel_size=3,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
            )
            self.layers.append(layer)
            in_ch = out_ch
        
        self.out_channels = out_ch
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, D, H, W]
        Returns:
            output: [B, 512, D/16, H/16, W/16] - Specific bottleneck features
            global_feat: [B, 512, 1, 1, 1] - Global pooled features for domain classifier
            skip_features: List of intermediate features for skip connections
        """
        skip_features = []
        for layer in self.layers:
            x = layer(x)
            skip_features.append(x)
        
        # Global average pooling for domain classifier
        global_feat = torch.mean(x, dim=[2, 3, 4], keepdim=True)  # [B, 512, 1, 1, 1]
        
        return x, global_feat, skip_features


class CompositionalLayer(nn.Module):
    """
    Residual Fusion Layer: combines shared and specific features at bottleneck
    
    Formula: fused = f_shared + Conv(concat(f_shared, f_specific))
    
    This corresponds to the fusion method in DualNet_SS paper.
    Fusion happens at bottleneck level with 512 channels.
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        spatial_dims: int = 3,
        norm: str = "INSTANCE",
        act: str = "RELU",
    ):
        super().__init__()
        
        # Fusion convolution: concat [512, 512] -> 512
        self.fusion_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels * 2,  # concat shared + specific
            out_channels=in_channels,
            kernel_size=3,
            strides=1,
            act=act,
            norm=norm,
        )
    
    def forward(self, f_shared: torch.Tensor, f_specific: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_shared: [B, 512, D/16, H/16, W/16] - Shared bottleneck features
            f_specific: [B, 512, D/16, H/16, W/16] - Specific bottleneck features
        Returns:
            [B, 512, D/16, H/16, W/16] - Fused bottleneck features
        """
        # Concatenate shared and specific features
        concat_feat = torch.cat([f_shared, f_specific], dim=1)  # [B, 1024, D/16, H/16, W/16]
        
        # Compute residual
        residual = self.fusion_conv(concat_feat)  # [B, 512, D/16, H/16, W/16]
        
        # Residual connection: fused = shared + residual
        fused = f_shared + residual
        
        return fused


class Decoder(nn.Module):
    """UNet decoder with skip connections"""
    
    def __init__(
        self,
        spatial_dims: int = 3,
        channels: Sequence[int] = (512, 256, 128, 64, 32),
        strides: Sequence[int] = (2, 2, 2, 2),
        skip_channels: Sequence[int] = None,
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "RELU",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        if skip_channels is None:
            skip_channels = [channels[i+1] for i in range(len(channels)-1)]
        
        for i in range(len(channels) - 1):
            # Upsample
            up = UpSample(
                spatial_dims=spatial_dims,
                in_channels=channels[i],
                out_channels=channels[i+1],
                scale_factor=strides[i],
                mode="nontrainable",
            )
            self.up_layers.append(up)
            
            # Conv after concatenation
            if i < len(skip_channels):
                conv_in_channels = channels[i+1] + skip_channels[i]
            else:
                conv_in_channels = channels[i+1]
            
            conv = ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=conv_in_channels,
                out_channels=channels[i+1],
                strides=1,
                kernel_size=3,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
            )
            self.conv_layers.append(conv)
    
    def forward(
        self,
        x: torch.Tensor,
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: bottleneck feature
            skip_features: features from encoder (reversed order)
        """
        for i, (up, conv) in enumerate(zip(self.up_layers, self.conv_layers)):
            x = up(x)
            if i < len(skip_features):
                x = torch.cat([x, skip_features[i]], dim=1)
            x = conv(x)
        return x


@register_model("unet_multimodal_deepfusion")
class MultimodalUNetDeepFusion(nn.Module):
    """
    Multi-modal UNet with Shared-Specific Deep Feature Fusion
    
    Architecture (following DualNet_SS fusion strategy with UNet3D encoders):
        1. Shared Encoder: Full UNet3D encoder (5 layers) for all modalities
        2. Specific Encoders: 4 independent full UNet3D encoders (one per modality)
        3. Residual Fusion: Combine shared + specific at bottleneck level
        4. Decoder: Skip connections from shared encoder → segmentation output
        5. Domain Classifier (optional): Auxiliary task using specific bottleneck features
    
    Config example:
        model:
          name: unet_multimodal_deepfusion
          num_modalities: 4
          num_classes: 3
          channels: [32, 64, 128, 256, 512]
          strides: [2, 2, 2, 2]
          num_res_units: 2
          norm: "INSTANCE"
          act: "RELU"
          dropout: 0.0
          domain_classifier:
            enabled: true
            loss_weight: 0.1  # Set to 0 to disable
    """
    
    def __init__(self, cfg: DictConfig | Dict[str, Any]):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        
        logger = get_logger()
        
        # Parse config
        num_modalities = int(get_config(cfg, "num_modalities", 4))
        num_classes = int(get_config(cfg, "num_classes", 3))
        spatial_dims = int(get_config(cfg, "spatial_dims", 3))
        
        channels = list(get_config(cfg, "channels", [32, 64, 128, 256, 512]))
        strides = list(get_config(cfg, "strides", [2, 2, 2, 2]))
        num_res_units = int(get_config(cfg, "num_res_units", 2))
        act = get_config(cfg, "act", "RELU")
        norm = get_config(cfg, "norm", "INSTANCE")
        dropout = float(get_config(cfg, "dropout", 0.0))
        
        # Domain classifier config
        domain_cfg = get_config(cfg, "domain_classifier", {})
        self.domain_enabled = bool(get_config(domain_cfg, "enabled", True))
        self.domain_loss_weight = float(get_config(domain_cfg, "loss_weight", 0.1))
        
        # Architecture configuration for deep fusion
        encoder_channels = channels  # [32, 64, 128, 256, 512]
        encoder_strides = strides + [1]  # [2, 2, 2, 2, 1] - last is bottleneck
        fusion_channels = channels[-1]  # 512 - bottleneck channels
        
        # Decoder configuration
        decoder_channels = list(reversed(channels))  # [512, 256, 128, 64, 32]
        decoder_strides = strides  # [2, 2, 2, 2]
        # Skip connections matching:
        # Encoder strides [2, 2, 2, 2, 1] means:
        # Layer 1: 1/2, Layer 2: 1/4, Layer 3: 1/8, Layer 4: 1/16, Layer 5: 1/16 (Bottleneck)
        # Decoder upsamples: 1/16->1/8, 1/8->1/4, 1/4->1/2, 1/2->1
        # Skips needed: Skip 3 (1/8), Skip 2 (1/4), Skip 1 (1/2)
        # Channels: channels[2]=128, channels[1]=64, channels[0]=32
        skip_channels = list(reversed(channels[:-2]))  # [128, 64, 32]
        
        self.num_modalities = num_modalities
        
        # 1. Shared encoder (full 5-layer UNet3D encoder)
        self.shared_encoder = SharedEncoder(
            spatial_dims=spatial_dims,
            channels=encoder_channels,
            strides=encoder_strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
        )
        
        # 2. Specific encoders (4 independent full UNet3D encoders)
        self.specific_encoders = nn.ModuleList([
            SpecificEncoder(
                spatial_dims=spatial_dims,
                channels=encoder_channels,
                strides=encoder_strides,
                num_res_units=num_res_units,
                norm=norm,
                act=act,
                dropout=dropout,
            )
            for _ in range(num_modalities)
        ])
        
        # 3. Residual fusion layer (at bottleneck)
        self.fusion_layer = CompositionalLayer(
            in_channels=fusion_channels,  # 512
            spatial_dims=spatial_dims,
            norm=norm,
            act=act,
        )
        
        # 3.5 Bottleneck reduction: 4 fused modalities [B, 2048, ...] -> [B, 512, ...]
        # This reduces the concatenated fused features to match decoder input
        self.bottleneck_reduce = nn.Conv3d(
            in_channels=fusion_channels * num_modalities,  # 2048 (512 × 4)
            out_channels=fusion_channels,  # 512
            kernel_size=1,
            bias=False,
        )
        logger.info(
            f"[MultimodalUNetDeepFusion] Bottleneck reduction: "
            f"{fusion_channels * num_modalities} -> {fusion_channels}"
        )
        
        # 4. Domain classifier (optional)
        if self.domain_enabled:
            # Input: global pooled specific features [B*4, 512, 1, 1, 1]
            # Direct classification without projection (bottleneck already 512)
            self.domain_classifier = nn.Linear(fusion_channels, num_modalities)
            logger.info(
                f"[MultimodalUNetDeepFusion] Domain classifier enabled "
                f"(loss_weight={self.domain_loss_weight})"
            )
        
        # 5. Decoder
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            channels=decoder_channels,
            strides=decoder_strides,
            skip_channels=skip_channels,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
        )
        
        # 7. Output convolution
        self.out_conv = nn.Conv3d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )
        
        logger.info(
            f"[MultimodalUNetDeepFusion] Created: "
            f"modalities={num_modalities}, spatial_dims={spatial_dims}, "
            f"out={num_classes}, channels={channels}, strides={encoder_strides}, "
            f"fusion=deep_bottleneck, res_units={num_res_units}, "
            f"act={act}, norm={norm}, dropout={dropout}"
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_domain_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 4, D, H, W] - Input with 4 modalities
            return_domain_logits: Whether to return domain classification logits
                                 (for training with domain loss)
        Returns:
            If return_domain_logits=False:
                logits: [B, 3, D, H, W] - Segmentation logits
            If return_domain_logits=True:
                (logits, domain_logits): 
                    logits: [B, 3, D, H, W]
                    domain_logits: [B*4, 4] - Domain classification logits
        """
        B, C, D, H, W = x.shape
        assert C == self.num_modalities, f"Expected {self.num_modalities} modalities, got {C}"
        
        # Step 1: Shared encoder (full 5-layer) - batch processing
        # Reshape: [B, 4, D, H, W] -> [B*4, 1, D, H, W]
        x_flat = x.view(B * C, 1, D, H, W)
        shared_feat, shared_global, shared_skips = self.shared_encoder(x_flat)
        # shared_feat: [B*4, 512, D/16, H/16, W/16] - Bottleneck
        # shared_global: [B*4, 512, 1, 1, 1] - Global pooled (unused for domain classifier)
        # shared_skips: List of [B*4, C_i, ...] - Skip features from all 5 layers
        
        # Step 2: Specific encoders (full 5-layer) - independent processing
        modalities = torch.split(x, 1, dim=1)  # 4 × [B, 1, D, H, W]
        
        specific_feats = []
        specific_globals = []
        all_specific_skips = []
        
        for encoder, modal in zip(self.specific_encoders, modalities):
            feat, global_feat, skips = encoder(modal)
            # feat: [B, 512, D/16, H/16, W/16] - Bottleneck
            # global_feat: [B, 512, 1, 1, 1] - Global pooled for domain classifier
            specific_feats.append(feat)
            specific_globals.append(global_feat)
            all_specific_skips.append(skips)
        
        # Step 3: Residual fusion at bottleneck for each modality
        # Extract shared bottleneck features for each modality
        shared_feats_per_modality = [
            shared_feat[i::C] for i in range(C)  # [B, 512, ...]
        ]
        
        fused_feats = []
        for shared_m, specific_m in zip(shared_feats_per_modality, specific_feats):
            fused_m = self.fusion_layer(shared_m, specific_m)  # [B, 512, ...]
            fused_feats.append(fused_m)
        
        # Concatenate fused features: 4 × [B, 512, ...] -> [B, 2048, ...]
        fused_concat = torch.cat(fused_feats, dim=1)  # [B, 2048, D/16, H/16, W/16]
        
        # Reduce to match decoder input: [B, 2048, ...] -> [B, 512, ...]
        fused_bottleneck = self.bottleneck_reduce(fused_concat)  # [B, 512, D/16, H/16, W/16]
        
        # Step 4: Prepare skip connections from shared encoder
        # shared_skips: [layer1, layer2, layer3, layer4, layer5(bottleneck)]
        # We need skips from layers 1-4 (not bottleneck) for decoder
        # Reshape shared_skips from [B*4, C, ...] to [B, 4, C, ...] and average
        fused_shared_skips = []
        for i, shared_skip in enumerate(shared_skips[:-2]):  # Exclude bottleneck
            # Reshape: [B*4, C_skip, ...] -> [B, 4, C_skip, ...]
            skip_shape = shared_skip.shape
            skip_reshaped = shared_skip.view(B, self.num_modalities, *skip_shape[1:])
            # Average across modalities
            skip_fused = skip_reshaped.mean(dim=1)  # [B, C_skip, ...]
            fused_shared_skips.append(skip_fused)
        
        # Skip order for decoder: [Layer4, Layer3, Layer2, Layer1]
        # Reverse the list
        skip_features = fused_shared_skips[::-1]
        
        # Step 5: Decode
        # Use fused_bottleneck as input (already reduced to 512 channels)
        decoded = self.decoder(fused_bottleneck, skip_features)
        
        # Step 6: Output segmentation logits
        logits = self.out_conv(decoded)
        
        # Step 7: Domain classification (if requested)
        if return_domain_logits and self.domain_enabled and self.domain_loss_weight > 0:
            # Use specific global features for domain classification
            # Concatenate: 4 × [B, 512, 1, 1, 1] -> [B*4, 512, 1, 1, 1]
            specific_global_concat = torch.cat(specific_globals, dim=0)  # [B*4, 512, 1, 1, 1]
            
            # Flatten and classify (no projection needed, already 512 dims)
            domain_feat = specific_global_concat.squeeze(-1).squeeze(-1).squeeze(-1)  # [B*4, 512]
            
            # Classify
            domain_logits = self.domain_classifier(domain_feat)  # [B*4, 4]
            
            return logits, domain_logits
        
        return logits
    
    def get_domain_loss_weight(self) -> float:
        """Get current domain loss weight (can be 0 to disable)"""
        if not self.domain_enabled:
            return 0.0
        return self.domain_loss_weight
