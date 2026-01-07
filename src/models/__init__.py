"""Model package initialization and registration."""

from ..registry import register_model

from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .densenet import (
    DenseNetSimple,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    DenseNet161,
)
from .vit import (
    ViTSimple,
    ViT_B_16,
    ViT_B_32,
    ViT_L_16,
    ViT_L_32,
    ViT_H_14,
)

from .efficientnet import (
    EfficientNet,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2S, EfficientNetV2M, EfficientNetV2L,
)

from .unet import UNet

# Register models with the unified registry
register_model('resnet18')(ResNet18)
register_model('resnet34')(ResNet34)
register_model('resnet50')(ResNet50)
register_model('resnet101')(ResNet101)
register_model('resnet152')(ResNet152)

register_model('densenet121')(DenseNet121)
register_model('densenet169')(DenseNet169)
register_model('densenet201')(DenseNet201)
register_model('densenet161')(DenseNet161)

register_model('vit_b_16')(ViT_B_16)
register_model('vit_b_32')(ViT_B_32)
register_model('vit_l_16')(ViT_L_16)
register_model('vit_l_32')(ViT_L_32)
register_model('vit_h_14')(ViT_H_14)

register_model('efficientnet_b0')(EfficientNetB0)
register_model('efficientnet_b1')(EfficientNetB1)
register_model('efficientnet_b2')(EfficientNetB2)
register_model('efficientnet_b3')(EfficientNetB3)
register_model('efficientnet_b4')(EfficientNetB4)
register_model('efficientnet_b5')(EfficientNetB5)
register_model('efficientnet_b6')(EfficientNetB6)
register_model('efficientnet_b7')(EfficientNetB7)
register_model('efficientnet_v2_s')(EfficientNetV2S)
register_model('efficientnet_v2_m')(EfficientNetV2M)
register_model('efficientnet_v2_l')(EfficientNetV2L)

__all__ = [
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
    'DenseNetSimple', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
    'ViTSimple', 'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14',
    "EfficientNet","EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
    "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
    "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L",
    "UNet",
]