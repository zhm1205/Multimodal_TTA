# src/datasets/hecktor21_nifti.py
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

from ..utils.logger import get_logger
from ..utils.config import require_config, get_config
from ..registry import register_dataset_builder
from .base_builder import BaseDatasetBuilder
from .transforms import get_seg_transforms


# ----------------------------
# Helpers
# ----------------------------
def _load_nifti_xyz_as_canonical(path: str, dtype=np.float32) -> np.ndarray:
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(dtype=dtype)  # (X,Y,Z)


def _resolve_path(path: Any, root_dir: Optional[str]) -> str:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return ""
    p = str(path)
    if root_dir and (not os.path.isabs(p)):
        return os.path.join(root_dir, p)
    return p


def _validate_shape(arr: np.ndarray, expected_shape: Optional[Tuple[int, int, int]], what: str, case_id: str) -> None:
    if expected_shape is None:
        return
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"[HECKTOR21] Shape mismatch for {what} case={case_id}: "
            f"got {tuple(arr.shape)}, expected {tuple(expected_shape)}. "
            f"This dataset assumes OFFLINE preprocessing already unified shape."
        )


def _to_binary_mask(y: np.ndarray) -> np.ndarray:
    # 常见：{0,1} / {0,255} / 其他非零
    if y.dtype.kind not in ("i", "u"):
        y = np.rint(y)
    y = y.astype(np.int16, copy=False)
    uniq = np.unique(y)
    if uniq.size == 0:
        return y.astype(np.uint8, copy=False)
    if np.all(np.isin(uniq, [0, 1])):
        return y.astype(np.uint8, copy=False)
    if np.all(np.isin(uniq, [0, 255])):
        return (y // 255).astype(np.uint8, copy=False)
    return (y != 0).astype(np.uint8, copy=False)


def _sample_val_indices_per_center(
    df_non_target: pd.DataFrame,
    center_code_col: str,
    val_per_center: int,
    seed: int,
) -> np.ndarray:
    if val_per_center <= 0 or len(df_non_target) == 0:
        return np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed)
    val_indices: List[int] = []

    # 为了稳定性：按 center 排序
    centers = sorted(df_non_target[center_code_col].astype(str).str.upper().unique().tolist())
    for c in centers:
        d = df_non_target[df_non_target[center_code_col].astype(str).str.upper() == c]
        idxs = d.index.to_numpy()
        if idxs.size == 0:
            continue
        k = min(val_per_center, int(idxs.size))
        chosen = rng.choice(idxs, size=k, replace=False)
        val_indices.extend(chosen.tolist())

    return np.array(val_indices, dtype=np.int64)


# ----------------------------
# Dataset
# ----------------------------
class Hecktor21Dataset(Dataset):
    """
    单一 manifest.csv + 动态 split：

    给定 target_center = T:
      - test  = center_code == T  的全部
      - train/val = center_code != T 的全部
        * 每个非 target center 抽 val_per_center 个做 val
        * 剩余做 train

    返回：
      image: FloatTensor [2, D, H, W]  (CT, PET)
      label: FloatTensor [1, D, H, W]  (binary GTVt)
      domain: center_code
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str,
        *,
        target_center: str,
        val_per_center: int = 5,
        split_seed: int = 2026,
        expected_shape: Optional[Tuple[int, int, int]] = None,  # (X,Y,Z) in nib array
        drop_unlabeled: bool = True,
        strict_label_values: bool = True,
        root_dir: Optional[str] = None,
        # columns
        patient_col: str = "patient_id",
        status_col: str = "status",
        ok_status_values: Sequence[str] = ("ok",),
        ct_col: str = "ct_proc",
        pt_col: str = "pt_proc",
        label_col: str = "gtvt_proc",
        center_code_col: str = "center_code",
        center_id_col: str = "center_id",
        # transforms
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.split = str(split).lower().strip()
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"[HECKTOR21] split must be in {{train,val,test}}, got '{split}'")

        self.expected_shape = expected_shape
        self.drop_unlabeled = bool(drop_unlabeled)
        self.strict_label_values = bool(strict_label_values)
        self.root_dir = root_dir
        self.transform = transform

        self.patient_col = patient_col
        self.status_col = status_col
        self.ok_status_values = [str(x).lower() for x in list(ok_status_values)]
        self.ct_col = ct_col
        self.pt_col = pt_col
        self.label_col = label_col
        self.center_code_col = center_code_col
        self.center_id_col = center_id_col

        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"[HECKTOR21] manifest_csv not found: {manifest_csv}")
        df = pd.read_csv(manifest_csv)

        # minimal required columns
        for c in [patient_col, ct_col, pt_col, center_code_col]:
            if c not in df.columns:
                raise ValueError(f"[HECKTOR21] manifest missing required column '{c}'")

        # ensure label/status columns exist
        if label_col not in df.columns:
            df[label_col] = np.nan
        if status_col not in df.columns:
            df[status_col] = "ok"  # 没有就默认全 ok（你也可以改成直接不做 status filter）

        # 1) status filter
        ok_set = set(self.ok_status_values)
        df = df[df[status_col].astype(str).str.lower().isin(ok_set)].copy()

        # 2) drop_unlabeled
        if self.drop_unlabeled:
            df = df[df[label_col].notna() & (df[label_col].astype(str) != "")].copy()

        # normalize center codes
        df[center_code_col] = df[center_code_col].astype(str).str.upper()

        target_center = str(target_center).upper().strip()
        if target_center == "":
            raise ValueError("[HECKTOR21] target_center cannot be empty")

        df_target = df[df[center_code_col] == target_center].copy()
        df_non_target = df[df[center_code_col] != target_center].copy()

        if len(df_target) == 0:
            raise ValueError(
                f"[HECKTOR21] target_center='{target_center}' has 0 samples after filtering. "
                f"Check center_code values in manifest."
            )
        if len(df_non_target) == 0:
            raise ValueError("[HECKTOR21] non-target set is empty; cannot build train/val.")

        # 3) sample val per non-target center
        val_indices = _sample_val_indices_per_center(
            df_non_target=df_non_target,
            center_code_col=center_code_col,
            val_per_center=int(val_per_center),
            seed=int(split_seed),
        )
        df_val = df_non_target.loc[val_indices].copy() if val_indices.size > 0 else df_non_target.iloc[0:0].copy()
        df_train = df_non_target.drop(index=val_indices).copy() if val_indices.size > 0 else df_non_target.copy()

        # 4) choose split df
        if self.split == "test":
            d_use = df_target
        elif self.split == "val":
            d_use = df_val
        else:
            d_use = df_train

        if len(d_use) == 0:
            raise ValueError(
                f"[HECKTOR21] split='{self.split}' becomes empty. "
                f"target_center={target_center}, val_per_center={val_per_center}."
            )

        # build index
        self._rows: List[Dict[str, Any]] = []
        for _, row in d_use.iterrows():
            self._rows.append(row.to_dict())

        self.logger.info(
            f"[HECKTOR21] split='{self.split}' n={len(self._rows)} | "
            f"target_center={target_center} | "
            f"non_target_centers={df_non_target[center_code_col].nunique()} | "
            f"val_per_center={val_per_center} seed={split_seed}"
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self._rows[idx]
        case_id = str(info.get(self.patient_col))

        center_code = str(info.get(self.center_code_col, "")).upper()
        center_id = info.get(self.center_id_col, None)
        try:
            center_id = int(center_id) if (center_id is not None and str(center_id) != "nan") else -1
        except Exception:
            center_id = -1

        ct_path = _resolve_path(info.get(self.ct_col, ""), self.root_dir)
        pt_path = _resolve_path(info.get(self.pt_col, ""), self.root_dir)
        lb_path = _resolve_path(info.get(self.label_col, ""), self.root_dir)

        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"[HECKTOR21] Missing CT file: {ct_path} (case={case_id})")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"[HECKTOR21] Missing PET file: {pt_path} (case={case_id})")
        if self.drop_unlabeled and ((not lb_path) or (not os.path.exists(lb_path))):
            raise FileNotFoundError(f"[HECKTOR21] Missing label file: {lb_path} (case={case_id})")

        # load (X,Y,Z)
        ct = _load_nifti_xyz_as_canonical(ct_path, dtype=np.float32)
        pt = _load_nifti_xyz_as_canonical(pt_path, dtype=np.float32)
        _validate_shape(ct, self.expected_shape, "ct", case_id)
        _validate_shape(pt, self.expected_shape, "pt", case_id)

        image = np.stack([ct, pt], axis=0).astype(np.float32, copy=False)  # [2,X,Y,Z]
        image_t = torch.from_numpy(image).float().permute(0, 3, 2, 1)       # [2,Z,Y,X] => [2,D,H,W]

        if (not lb_path) or (not os.path.exists(lb_path)):
            y_np = np.zeros(ct.shape, dtype=np.uint8)
        else:
            y = _load_nifti_xyz_as_canonical(lb_path, dtype=np.float32)
            _validate_shape(y, self.expected_shape, "label", case_id)
            y_np = _to_binary_mask(y)

        y_t = torch.from_numpy(y_np.astype(np.float32, copy=False)).permute(2, 1, 0).unsqueeze(0)  # [1,Z,Y,X]

        if self.strict_label_values:
            uniq = torch.unique(y_t).detach().cpu().tolist()
            bad = [v for v in uniq if v not in (0.0, 1.0)]
            if bad:
                raise ValueError(f"[HECKTOR21] Label must be binary {{0,1}}. got={uniq} (case={case_id})")

        if self.transform is not None:
            out = self.transform(image_t, y_t)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image_t, y_t = out
            else:
                raise RuntimeError("[HECKTOR21] transform must return (image, label).")

        return {
            "image": image_t,           # [2,D,H,W]
            "label": y_t,               # [1,D,H,W]
            "case_id": case_id,
            "domain": center_code,      # 直接用 center_code 当 domain
            "center_code": center_code,
            "center_id": center_id,
            "index": int(idx),
        }


# ----------------------------
# Builder
# ----------------------------
@register_dataset_builder("hecktor21")
class Hecktor21Builder(BaseDatasetBuilder):
    """
    最简配置（只需要一个 manifest）：

    dataset:
      name: hecktor21
      manifest_csv: /path/to/manifest.csv
      expected_shape: [144,144,48]        # 这是 (X,Y,Z)，要和预处理输出一致
      drop_unlabeled: true
      strict_label_values: true

      patient_col: patient_id
      status_col: status
      ok_status_values: ["ok"]
      ct_col: ct_proc
      pt_col: pt_proc
      label_col: gtvt_proc
      center_code_col: center_code
      center_id_col: center_id
      root_dir: null                      # 可选：如果 manifest 里是相对路径

      # 动态切 target 的关键参数
      target_center: "CHUS"
      val_per_center: 5
      split_seed: 2026
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        self.manifest_csv = str(require_config(dcfg, "manifest_csv", type_=str))

        exp_shape = get_config(dcfg, "expected_shape", None)
        self.expected_shape = tuple(exp_shape) if exp_shape is not None else None

        self.drop_unlabeled = bool(get_config(dcfg, "drop_unlabeled", True))
        self.strict_label_values = bool(get_config(dcfg, "strict_label_values", True))

        self.patient_col = str(get_config(dcfg, "patient_col", "patient_id"))
        self.status_col = str(get_config(dcfg, "status_col", "status"))
        self.ok_status_values = list(get_config(dcfg, "ok_status_values", ["ok"]))

        self.ct_col = str(get_config(dcfg, "ct_col", "ct_proc"))
        self.pt_col = str(get_config(dcfg, "pt_col", "pt_proc"))
        self.label_col = str(get_config(dcfg, "label_col", "gtvt_proc"))

        self.center_code_col = str(get_config(dcfg, "center_code_col", "center_code"))
        self.center_id_col = str(get_config(dcfg, "center_id_col", "center_id"))
        self.root_dir = get_config(dcfg, "root_dir", None)

        # dynamic split
        self.target_center = str(require_config(dcfg, "target_center", type_=str))
        self.val_per_center = int(get_config(dcfg, "val_per_center", 5))
        self.split_seed = int(get_config(dcfg, "split_seed", 2026))

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split_norm = self._normalize_split(split)

        # transforms（沿用你工程的 get_seg_transforms）
        transform = overrides.get("transform", None)
        if transform is None:
            dcfg: DictConfig = require_config(self.config, "training.data")
            tcfg: DictConfig = get_config(dcfg, "transforms", DictConfig({}))

            normalize = bool(require_config(tcfg, "normalize"))
            geom_aug = bool(require_config(tcfg, "geom_aug"))
            intensity_aug = bool(require_config(tcfg, "intensity_aug"))

            mean = get_config(tcfg, "mean", [0.0, 0.0])
            std = get_config(tcfg, "std", [1.0, 1.0])

            intensity_policy = get_config(tcfg, "intensity_policy", None)

            image_size = get_config(tcfg, "image_size", None)
            if image_size is not None:
                if len(list(image_size)) != 3:
                    raise ValueError(f"[hecktor21] training.data.transforms.image_size must be [D,H,W]")
                image_size = [int(x) for x in list(image_size)]

            transform = get_seg_transforms(
                ndim=3,
                split=split_norm,
                normalize=normalize,
                geom_aug=geom_aug,
                intensity_aug=intensity_aug,
                mean=mean,
                std=std,
                expected_label_channels=1,
                region_label_as_float=True,
                image_size=image_size,  # 仅形状校验，不做 resize
                intensity_policy=intensity_policy,
                channel_names=["ct", "pt"],
            )

        expected_shape = overrides.get("expected_shape", self.expected_shape)
        strict_label_values = bool(overrides.get("strict_label_values", self.strict_label_values))

        ds = Hecktor21Dataset(
            manifest_csv=self.manifest_csv,
            split=split_norm,
            target_center=str(overrides.get("target_center", self.target_center)),
            val_per_center=int(overrides.get("val_per_center", self.val_per_center)),
            split_seed=int(overrides.get("split_seed", self.split_seed)),
            expected_shape=expected_shape,
            drop_unlabeled=bool(overrides.get("drop_unlabeled", self.drop_unlabeled)),
            strict_label_values=strict_label_values,
            root_dir=overrides.get("root_dir", self.root_dir),

            patient_col=self.patient_col,
            status_col=self.status_col,
            ok_status_values=self.ok_status_values,
            ct_col=self.ct_col,
            pt_col=self.pt_col,
            label_col=self.label_col,
            center_code_col=self.center_code_col,
            center_id_col=self.center_id_col,

            transform=transform,
            logger=self.logger,
        )
        return ds