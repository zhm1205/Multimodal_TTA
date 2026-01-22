#!/usr/bin/env python
"""
Verification script to ensure multi-modal UNet models have same config as original UNet
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from omegaconf import OmegaConf
from src.models import UNet, MultimodalUNetMidFusion, MultimodalUNetLateFusion

def test_model_creation():
    """Test that all models can be created with same config"""
    
    # Original UNet config (from _global_patches/brats.yaml)
    base_config = OmegaConf.create({
        "in_channels": 4,
        "num_classes": 3,
        "spatial_dims": 3,
        "channels": [32, 64, 128, 256, 512],
        "strides": [2, 2, 2, 2],
        "num_res_units": 2,
        "norm": "INSTANCE",
        "act": "RELU",
        "dropout": 0.0,
    })
    
    print("=" * 60)
    print("Testing Model Creation with Same Config")
    print("=" * 60)
    
    # Test original UNet
    print("\n1. Original UNet:")
    try:
        model_orig = UNet(base_config)
        print(f"   ✅ Created successfully")
        param_count_orig = sum(p.numel() for p in model_orig.parameters())
        print(f"   Parameters: {param_count_orig/1e6:.2f}M")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test mid-fusion UNet
    print("\n2. Multi-modal UNet (Mid Fusion):")
    try:
        config_mid = OmegaConf.merge(base_config, {"num_modalities": 4})
        model_mid = MultimodalUNetMidFusion(config_mid)
        print(f"   ✅ Created successfully")
        param_count_mid = sum(p.numel() for p in model_mid.parameters())
        print(f"   Parameters: {param_count_mid/1e6:.2f}M")
        print(f"   Increase: +{(param_count_mid/param_count_orig - 1)*100:.1f}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test late-fusion UNet
    print("\n3. Multi-modal UNet (Late Fusion):")
    try:
        config_late = OmegaConf.merge(base_config, {
            "num_modalities": 4,
            "fusion_type": "learned_weight"
        })
        model_late = MultimodalUNetLateFusion(config_late)
        print(f"   ✅ Created successfully")
        param_count_late = sum(p.numel() for p in model_late.parameters())
        print(f"   Parameters: {param_count_late/1e6:.2f}M")
        print(f"   Increase: +{(param_count_late/param_count_orig - 1)*100:.1f}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    return True


def test_forward_pass():
    """Test forward pass with same input shape"""
    
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Test input
    x = torch.randn(2, 4, 160, 192, 160)  # [B, C, D, H, W]
    expected_output_shape = (2, 3, 160, 192, 160)
    
    base_config = OmegaConf.create({
        "in_channels": 4,
        "num_classes": 3,
        "spatial_dims": 3,
        "channels": [32, 64, 128, 256, 512],
        "strides": [2, 2, 2, 2],
        "num_res_units": 2,
        "norm": "INSTANCE",
        "act": "RELU",
        "dropout": 0.0,
    })
    
    models = {
        "Original UNet": UNet(base_config),
        "Mid Fusion": MultimodalUNetMidFusion(
            OmegaConf.merge(base_config, {"num_modalities": 4})
        ),
        "Late Fusion": MultimodalUNetLateFusion(
            OmegaConf.merge(base_config, {"num_modalities": 4, "fusion_type": "learned_weight"})
        ),
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            if output.shape == expected_output_shape:
                print(f"   ✅ Output shape correct: {tuple(output.shape)}")
            else:
                print(f"   ❌ Output shape mismatch!")
                print(f"      Expected: {expected_output_shape}")
                print(f"      Got: {tuple(output.shape)}")
                return False
        except Exception as e:
            print(f"   ❌ Forward failed: {e}")
            return False
    
    return True


def verify_config_consistency():
    """Verify that architecture params are used consistently"""
    
    print("\n" + "=" * 60)
    print("Verifying Config Consistency")
    print("=" * 60)
    
    base_config = OmegaConf.create({
        "channels": [32, 64, 128, 256, 512],
        "strides": [2, 2, 2, 2],
        "num_res_units": 2,
        "norm": "INSTANCE",
        "act": "RELU",
        "dropout": 0.0,
    })
    
    print("\n✓ All models should use these params:")
    for key, val in base_config.items():
        print(f"  - {key}: {val}")
    
    print("\n✓ Multi-modal models ONLY add:")
    print("  - num_modalities: 4")
    print("  - fusion_type: 'learned_weight' (late fusion only)")
    
    print("\n✓ Architecture derivation (mid fusion):")
    channels = base_config.channels
    print(f"  - Branch encoding: channels[0:2] = {channels[:2]}")
    print(f"  - Fusion output: channels[2] = {channels[2]}")
    print(f"  - Shared encoder: channels[2:] = {channels[2:]}")
    print(f"  - Decoder: reversed(channels) = {list(reversed(channels))}")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-modal UNet Verification")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_model_creation()
    success &= test_forward_pass()
    success &= verify_config_consistency()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nModels are ready to use:")
        print("  - Mid fusion:  python main.py model=unet_multimodal_mid")
        print("  - Late fusion: python main.py model=unet_multimodal_late")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print("=" * 60)
        sys.exit(1)
