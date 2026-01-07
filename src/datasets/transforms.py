# file: src/utils/transforms.py
from __future__ import annotations

from typing import Callable, Tuple, Any, List, Sequence, Optional

import torch
from monai.transforms import (
    Compose,
    RandAxisFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

LabelTensor = torch.Tensor
ImageTensor = torch.Tensor


def _build_3d_seg_transforms(
    split: str,
    *,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    # 用于防止 silent bug（例如 label 误带 channel 维）
    # - None: 不做强校验（仅按维度自动判断 raw vs region）
    # - 0:    强制 label 是 raw（[D,H,W] 或 [1,D,H,W]）
    # - >0:   强制 label 是 region（[N,D,H,W] 且 N==expected_label_channels）
    expected_label_channels: Optional[int] = None,
    # region label 输出 dtype：训练 sigmoid multi-label 通常用 float32
    region_label_as_float: bool = True,
    # 仅做“尺寸一致性校验”，不做 resize/crop/pad。
    # 期望格式：[D,H,W]，例如 [160,196,160]
    image_size: Optional[Sequence[int]] = None,
) -> Callable[[ImageTensor, LabelTensor], Tuple[ImageTensor, LabelTensor]]:
    """
    3D segmentation transforms（仅一个 label 输入，按维度自动处理）：

    输入:
      image: [C, D, H, W] float32
      label:
        - raw label map: [D, H, W] 或 [1, D, H, W]
        - region masks:  [N, D, H, W]   (N=3 for ET/TC/WT)

    输出:
      image: [C, D, H, W] float32
      label:
        - raw:    [D, H, W] long
        - region: [N, D, H, W] float32 (默认) 或保持原 dtype（region_label_as_float=False）

    注意：
      - geom_aug / intensity_aug 仅在 train split 生效
      - 这里的几何增强只用 Flip / Rotate90，不涉及插值，对 label/region 都安全
      - normalize=True 时，最后做 per-channel (x-mean)/std
      - image_size 仅用于强校验，不会触发任何在线 resize/crop/pad
    """
    split = str(split).lower()
    is_train = split == "train"

    if not is_train:
        geom_aug = False
        intensity_aug = False

    # --------- expected spatial size (D,H,W) ----------
    expected_spatial: Optional[Tuple[int, int, int]] = None
    if image_size is not None:
        if len(image_size) != 3:
            raise ValueError(f"[3DTransforms] image_size must be [D,H,W], got {list(image_size)}")
        expected_spatial = (int(image_size[0]), int(image_size[1]), int(image_size[2]))

    xforms: List[Any] = []

    if geom_aug:
        xforms.extend(
            [
                RandAxisFlipd(keys=["image", "label"], prob=0.5),
                RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            ]
        )

    if intensity_aug:
        xforms.extend(
            [
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ]
        )

    monai_compose = Compose(xforms)

    def _normalize_img(img: torch.Tensor) -> torch.Tensor:
        if not normalize:
            return img

        if img.ndim != 4:
            raise ValueError(f"[3DTransforms] expect image [C,D,H,W], got {tuple(img.shape)}")

        c = int(img.shape[0])

        if mean is None:
            mean_t = torch.zeros(c, dtype=img.dtype, device=img.device)
        else:
            mean_t = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
        if mean_t.numel() == 1:
            mean_t = mean_t.repeat(c)
        if mean_t.numel() != c:
            raise RuntimeError(f"[3DTransforms] len(mean)={mean_t.numel()} != C={c}")

        if std is None:
            std_t = torch.ones(c, dtype=img.dtype, device=img.device)
        else:
            std_t = torch.as_tensor(std, dtype=img.dtype, device=img.device)
        if std_t.numel() == 1:
            std_t = std_t.repeat(c)
        if std_t.numel() != c:
            raise RuntimeError(f"[3DTransforms] len(std)={std_t.numel()} != C={c}")

        view_shape = (c,) + (1,) * (img.ndim - 1)  # [C,1,1,1]
        return (img - mean_t.view(view_shape)) / std_t.view(view_shape)

    def _infer_label_kind(lbl: torch.Tensor) -> str:
        """
        Returns: "raw" or "region"
        raw:    [D,H,W] or [1,D,H,W]
        region: [N,D,H,W] (N>=2, or N==expected_label_channels if set)
        """
        if lbl.ndim == 3:
            return "raw"
        if lbl.ndim == 4:
            n = int(lbl.shape[0])
            if expected_label_channels is not None and expected_label_channels > 0:
                return "region"
            return "raw" if n == 1 else "region"
        raise ValueError(f"[3DTransforms] label ndim must be 3 or 4, got {lbl.ndim}")

    def _check_spatial(name: str, t: torch.Tensor, spatial: Tuple[int, int, int]) -> None:
        # expect [*, D, H, W]
        if t.ndim < 3:
            raise ValueError(f"[3DTransforms] {name} must have at least 3 dims for spatial, got {tuple(t.shape)}")
        got = tuple(int(x) for x in t.shape[-3:])
        if got != spatial:
            raise ValueError(
                f"[3DTransforms] {name} spatial mismatch: got {got}, expected {spatial}. "
                f"This pipeline assumes OFFLINE preprocessing already fixed shapes; "
                f"no online resize/crop/pad is performed."
            )

    def _apply(image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---------- image checks ----------
        if image.ndim != 4:
            raise ValueError(f"[3DTransforms] expect image [C,D,H,W], got {tuple(image.shape)}")

        if expected_spatial is not None:
            _check_spatial("image", image, expected_spatial)

        # ---------- infer label kind ----------
        kind = _infer_label_kind(label)

        # ---------- optional hard checks to prevent silent bugs ----------
        if expected_label_channels is not None:
            if expected_label_channels == 0:
                # force raw
                if label.ndim == 4 and int(label.shape[0]) != 1:
                    raise ValueError(
                        f"[3DTransforms] expected raw label but got [N,D,H,W] with N={int(label.shape[0])}"
                    )
                kind = "raw"
            elif expected_label_channels > 0:
                # force region with exact channel count
                if label.ndim != 4:
                    raise ValueError(
                        f"[3DTransforms] expected region label [N,D,H,W] but got {tuple(label.shape)}"
                    )
                n = int(label.shape[0])
                if n != expected_label_channels:
                    raise ValueError(
                        f"[3DTransforms] expected region channels N={expected_label_channels}, got N={n}"
                    )
                kind = "region"

        # ---------- prepare label for dict transforms ----------
        # For dict transforms, keep channel-first convention:
        # - raw:    [D,H,W] -> [1,D,H,W]
        # - region: [N,D,H,W] keep as-is
        if kind == "raw":
            if label.ndim == 3:
                label_in = label.unsqueeze(0)  # [1,D,H,W]
            else:
                label_in = label
                if int(label_in.shape[0]) != 1:
                    raise ValueError(f"[3DTransforms] raw label expects N=1, got {int(label_in.shape[0])}")
        else:
            if label.ndim != 4:
                raise ValueError(f"[3DTransforms] region label expects [N,D,H,W], got {tuple(label.shape)}")
            label_in = label

        if expected_spatial is not None:
            _check_spatial("label", label_in, expected_spatial)

        data = {"image": image, "label": label_in}

        out = monai_compose(data) if len(xforms) > 0 else data
        img: torch.Tensor = out["image"]
        lbl: torch.Tensor = out["label"]

        # after aug, shape should still be identical (flip/rot90 do not change shape)
        if expected_spatial is not None:
            _check_spatial("image(after)", img, expected_spatial)
            _check_spatial("label(after)", lbl, expected_spatial)

        # ---------- restore label shape & dtype ----------
        if kind == "raw":
            # [1,D,H,W] -> [D,H,W]
            if lbl.ndim != 4 or int(lbl.shape[0]) != 1:
                raise ValueError(
                    f"[3DTransforms] raw label after transform should be [1,D,H,W], got {tuple(lbl.shape)}"
                )
            lbl = lbl[0].long()
        else:
            # keep [N,D,H,W]
            if lbl.ndim != 4:
                raise ValueError(
                    f"[3DTransforms] region label after transform should be [N,D,H,W], got {tuple(lbl.shape)}"
                )
            if region_label_as_float:
                lbl = lbl.float()

        # ---------- normalize at the end ----------
        img = _normalize_img(img)

        return img, lbl

    return _apply


def get_seg_transforms(
    *,
    ndim: int,
    split: str,
    normalize: bool = True,
    geom_aug: bool = True,
    intensity_aug: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    expected_label_channels: Optional[int] = None,
    region_label_as_float: bool = True,
    # 仅用于尺寸一致性校验，不做 resize/crop/pad。
    # 直接用你现有配置 training.data.transforms.image_size: [D,H,W]
    image_size: Optional[Sequence[int]] = None,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    统一入口（当前仅支持 3D）：

    推荐用法（你现在的 ET/TC/WT 三通道训练）：
      expected_label_channels: 3
      region_label_as_float: true
      image_size: [160,196,160]  # 仅校验，不会触发 resize/crop/pad
    """
    if ndim != 3:
        raise ValueError(f"get_seg_transforms currently only supports 3D (ndim=3). Got ndim={ndim}")

    return _build_3d_seg_transforms(
        split=split,
        normalize=normalize,
        geom_aug=geom_aug,
        intensity_aug=intensity_aug,
        mean=mean,
        std=std,
        expected_label_channels=expected_label_channels,
        region_label_as_float=region_label_as_float,
        image_size=image_size,
    )