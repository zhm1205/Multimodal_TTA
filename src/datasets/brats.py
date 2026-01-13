"""
BraTS Multi-source NIfTI Dataset (LOCO-style target selection, single-pass loading)

Design (your latest requirement):
- Load ALL sources (each has its own processed.csv) at builder init.
- Ignore CSV 'split' completely.
- Choose ONE source as target_domain -> ALL its labeled cases go to test.
- All other sources -> all labeled cases go to source pool, then sample val_per_source per source for val;
  remaining go to train.
- Split assignment is computed ONCE in the Builder and cached to guarantee train/val disjointness and
  consistency across dataset construction.

Dataset returns:
  image : FloatTensor [C, D, H, W], C=4 for (t1n,t1c,t2w,t2f)
  label : FloatTensor [3, D, H, W] region masks in fixed order [ET,TC,WT]
  domain: source.name
  profile: source.profile
  case_id: str

CSV expected columns (minimum):
  subject_id, modality, img_path, label_path, split(optional)
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
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
# Region definition (fixed)
# ----------------------------
REGION_ORDER: List[str] = ["ET", "TC", "WT"]
EXPECTED_REGION_CHANNELS: int = 3


DEFAULT_REGION_MAPS: Dict[str, Dict[str, List[int]]] = {
    "gli": {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3]},
    "ssa": {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3]},
    "ped": {"ET": [1], "TC": [1, 2, 3], "WT": [1, 2, 3, 4]},
}


# ----------------------------
# Helpers
# ----------------------------
def _load_nifti_xyz_as_canonical(path: str, dtype=np.float32) -> np.ndarray:
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(dtype=dtype)


def _safe_round_label(label: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    if label.dtype.kind in ("i", "u"):
        return label.astype(np.int16, copy=False)
    rounded = np.rint(label)
    # tolerate; caller may do strict checks
    _ = tol
    return rounded.astype(np.int16, copy=False)


def _validate_shape(arr: np.ndarray, expected_shape: Optional[Tuple[int, int, int]], what: str, case_id: str) -> None:
    if expected_shape is None:
        return
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"[BraTS-LOCO] Shape mismatch for {what} case={case_id}: "
            f"got {tuple(arr.shape)}, expected {tuple(expected_shape)}. "
            f"This dataset assumes OFFLINE preprocessing already unified shape."
        )


def _resolve_path(path: Any, root_dir: Optional[str]) -> str:
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return ""
    p = str(path)
    if root_dir and (not os.path.isabs(p)):
        return os.path.join(root_dir, p)
    return p


def _build_region_masks_from_raw(
    y_id: torch.Tensor,  # [D,H,W] long
    region_map: Dict[str, Sequence[int]],
) -> torch.Tensor:
    masks: List[torch.Tensor] = []
    for rname in REGION_ORDER:
        vals = region_map.get(rname, [])
        if not vals:
            masks.append(torch.zeros_like(y_id, dtype=torch.float32))
            continue
        m = torch.zeros_like(y_id, dtype=torch.bool)
        for v in vals:
            m |= (y_id == int(v))
        masks.append(m.float())
    return torch.stack(masks, dim=0)  # [3,D,H,W]


# ----------------------------
# CSV parsing (case-level)
# ----------------------------
def _parse_processed_csv_to_cases(
    csv_path: str,
    modality_order: Sequence[str],
    *,
    root_dir: Optional[str],
    drop_unlabeled: bool,
    split_col: str = "split",
    subject_col: str = "subject_id",
    modality_col: str = "modality",
    img_col: str = "img_path",
    label_col: str = "label_path",
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """
    cases[case_id] = {
      "split": <csv_split_str> (kept but ignored later),
      "modalities": {mod: img_path},
      "label": label_path
    }
    """
    logger = logger or get_logger()
    df = pd.read_csv(csv_path)

    for c in [subject_col, modality_col, img_col]:
        if c not in df.columns:
            raise ValueError(f"[BraTS-LOCO] CSV missing required column '{c}': {csv_path}")

    if split_col not in df.columns:
        df[split_col] = "na"
    if label_col not in df.columns:
        df[label_col] = np.nan

    cases: Dict[str, Dict[str, Any]] = {}
    required_mods = [m.lower() for m in modality_order]

    for _, row in df.iterrows():
        case_id = str(row[subject_col])
        mod = str(row[modality_col]).strip().lower()
        split = str(row[split_col]).strip().lower()

        img_path = _resolve_path(row[img_col], root_dir)
        label_path = _resolve_path(row[label_col], root_dir) if pd.notna(row[label_col]) else ""

        if case_id not in cases:
            cases[case_id] = {"split": split, "modalities": {}, "label": label_path}
        else:
            # keep first
            if (not cases[case_id]["label"]) and label_path:
                cases[case_id]["label"] = label_path

        cases[case_id]["modalities"][mod] = img_path

    # filter invalid cases
    valid: Dict[str, Dict[str, Any]] = {}
    dropped_missing_mod = 0
    dropped_no_label = 0

    for case_id, info in cases.items():
        mods = info["modalities"]
        if any(m not in mods for m in required_mods):
            dropped_missing_mod += 1
            continue
        if drop_unlabeled and (not info.get("label")):
            dropped_no_label += 1
            continue
        valid[case_id] = info

    logger.info(
        f"[BraTS-LOCO] Parsed {csv_path}: total_cases={len(cases)}, valid_cases={len(valid)}, "
        f"dropped_missing_mod={dropped_missing_mod}, dropped_no_label={dropped_no_label}"
    )
    return valid


# ----------------------------
# Source spec
# ----------------------------
@dataclass(frozen=True)
class SourceSpec:
    name: str
    csv_path: str
    profile: str
    root_dir: Optional[str]
    region_map: Dict[str, List[int]]


def _stable_int_hash(s: str) -> int:
    # deterministic across runs/machines
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


# ----------------------------
# Dataset (takes prepared index list)
# ----------------------------
class BratsDataset(Dataset):
    """
    items: List[(src, case_id, info)]
    info: {"modalities": {mod: path}, "label": label_path, ...}
    """

    def __init__(
        self,
        items: List[Tuple[SourceSpec, str, Dict[str, Any]]],
        split: str,
        modality_order: Sequence[str],
        expected_shape: Optional[Tuple[int, int, int]],
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]],
        drop_unlabeled: bool,
        strict_label_values: bool,
        logger=None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.items = items
        self.split = str(split).lower()
        self.modality_order = [m.lower() for m in modality_order]
        self.expected_shape = expected_shape
        self.transform = transform
        self.drop_unlabeled = bool(drop_unlabeled)
        self.strict_label_values = bool(strict_label_values)

        if len(self.items) == 0:
            raise ValueError(f"[BraTS-LOCO] Empty dataset for split='{self.split}'.")

        self.logger.info(f"[BraTS-LOCO] Built dataset split='{self.split}' n={len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src, case_id, info = self.items[idx]

        # ---- load modalities ----
        img_list: List[np.ndarray] = []
        for mod in self.modality_order:
            p = info["modalities"][mod]
            if not os.path.exists(p):
                raise FileNotFoundError(f"[BraTS-LOCO] Missing image file: {p} (case={case_id}, mod={mod})")
            vol = _load_nifti_xyz_as_canonical(p, dtype=np.float32)  # [X,Y,Z]
            _validate_shape(vol, self.expected_shape, what=f"image/{mod}", case_id=case_id)
            img_list.append(vol)

        image = np.stack(img_list, axis=0).astype(np.float32, copy=False)  # [C,X,Y,Z]
        image_t = torch.from_numpy(image).float().permute(0, 3, 2, 1)      # [C,Z,Y,X] => [C,D,H,W]

        # ---- load label ----
        label_path = info.get("label", "")
        if (not label_path) or (not os.path.exists(label_path)):
            if self.drop_unlabeled:
                raise FileNotFoundError(f"[BraTS-LOCO] Missing label: {label_path} (case={case_id}, src={src.name})")
            y_np = np.zeros(image.shape[1:], dtype=np.int16)
        else:
            y = _load_nifti_xyz_as_canonical(label_path, dtype=np.float32)
            _validate_shape(y, self.expected_shape, what="label", case_id=case_id)
            y_np = _safe_round_label(y)

        y_id = torch.from_numpy(y_np.astype(np.int64, copy=False)).long().permute(2, 1, 0)  # [Z,Y,X]

        if self.strict_label_values:
            uniq = torch.unique(y_id).detach().cpu().tolist()
            bad = [v for v in uniq if (v < 0 or v > 20)]
            if bad:
                raise ValueError(
                    f"[BraTS-LOCO] Abnormal label values {bad} in case={case_id} src={src.name}. "
                    f"Likely non-nearest interpolation in preprocessing."
                )

        y_reg = _build_region_masks_from_raw(y_id, region_map=src.region_map)  # [3,D,H,W] float

        # ---- transform ----
        if self.transform is not None:
            out = self.transform(image_t, y_reg)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image_t, y_reg = out
            else:
                raise RuntimeError("[BraTS-LOCO] transform must return (image, label_region).")

        if y_reg.ndim != 4 or int(y_reg.shape[0]) != EXPECTED_REGION_CHANNELS:
            raise ValueError(f"[BraTS-LOCO] Label must be [3,D,H,W], got {tuple(y_reg.shape)} (case={case_id}).")

        return {
            "image": image_t,
            "label": y_reg,
            "case_id": case_id,
            "domain": src.name,
            "profile": src.profile,
            "split": self.split,
            "index": int(idx),
        }


# ----------------------------
# Builder / Registry
# ----------------------------
@register_dataset_builder("brats")
class BratsBuilder(BaseDatasetBuilder):
    """
    Config (simple):

    dataset:
      name: brats
      expected_shape: [160,192,160]
      drop_unlabeled: true
      strict_label_values: false

      target_domain: brats24_ssa
      val_per_source: 5
      seed: 2026

      # csv schema
      split_col: split
      subject_col: subject_id
      modality_col: modality
      img_col: img_path
      label_col: label_path

      sources:
        - name: brats25_glipre
          profile: gli
          csv_path: /.../BraTS25-GLIPRE/processed.csv
          root_dir: null
          region_map: {ET:[3], TC:[1,3], WT:[1,2,3]}

        - name: brats24_ssa
          profile: ssa
          csv_path: /.../BraTS23-SSA/processed.csv

        - name: brats24_ped
          profile: ped
          csv_path: /.../BraTS24-PED/processed.csv
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        # shape & basic flags
        exp_shape = get_config(dcfg, "expected_shape", None)
        self.expected_shape = tuple(exp_shape) if exp_shape is not None else None

        self.drop_unlabeled = bool(get_config(dcfg, "drop_unlabeled", True))
        self.strict_label_values = bool(get_config(dcfg, "strict_label_values", False))

        # LOCO control
        self.target_domain = str(require_config(dcfg, "target_domain", type_=str))
        self.val_per_source = int(get_config(dcfg, "val_per_source", 5))
        self.seed = int(get_config(dcfg, "seed", 2026))

        # csv schema (keep explicit)
        self.split_col = str(get_config(dcfg, "split_col", "split"))
        self.subject_col = str(get_config(dcfg, "subject_col", "subject_id"))
        self.modality_col = str(get_config(dcfg, "modality_col", "modality"))
        self.img_col = str(get_config(dcfg, "img_col", "img_path"))
        self.label_col = str(get_config(dcfg, "label_col", "label_path"))

        # sources
        sources_cfg = get_config(dcfg, "sources", None)
        if sources_cfg is None:
            raise ValueError("[BraTS-LOCO] 'dataset.sources' is required.")

        self.sources: List[SourceSpec] = []
        for sc in sources_cfg:
            sname = str(require_config(sc, "name", type_=str))
            csv_path = str(require_config(sc, "csv_path", type_=str))
            profile = str(get_config(sc, "profile", "gli")).lower()
            root_dir = get_config(sc, "root_dir", None)

            region_map = DEFAULT_REGION_MAPS.get(profile, DEFAULT_REGION_MAPS["gli"])
            region_map_override = get_config(sc, "region_map", None)
            if region_map_override is not None:
                region_map = {k: [int(x) for x in list(v)] for k, v in dict(region_map_override).items()}
            else:
                region_map = {k: [int(x) for x in list(v)] for k, v in dict(region_map).items()}

            self.sources.append(
                SourceSpec(
                    name=sname,
                    csv_path=csv_path,
                    profile=profile,
                    root_dir=root_dir,
                    region_map=region_map,
                )
            )

        # fixed modality order in this project
        self.modality_order = ("t1n", "t1c", "t2w", "t2f")

        # sanity: target must exist
        all_names = [s.name for s in self.sources]
        if self.target_domain not in all_names:
            raise ValueError(
                f"[BraTS-LOCO] target_domain='{self.target_domain}' not in sources={all_names}. "
                f"Set dataset.target_domain correctly."
            )

        # Load ALL cases once, then assign split once
        self._cases_by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._assigned: Dict[Tuple[str, str], str] = {}  # (source_name, case_id) -> train/val/test

        self._load_all_sources()
        self._assign_splits_once()

    def _load_all_sources(self) -> None:
        for src in self.sources:
            if not os.path.exists(src.csv_path):
                raise FileNotFoundError(f"[BraTS-LOCO] CSV not found: {src.csv_path}")

            cases = _parse_processed_csv_to_cases(
                csv_path=src.csv_path,
                modality_order=self.modality_order,
                root_dir=src.root_dir,
                drop_unlabeled=self.drop_unlabeled,
                split_col=self.split_col,
                subject_col=self.subject_col,
                modality_col=self.modality_col,
                img_col=self.img_col,
                label_col=self.label_col,
                logger=self.logger,
            )
            self._cases_by_source[src.name] = cases

        # log counts
        for sname, cases in self._cases_by_source.items():
            self.logger.info(f"[BraTS-LOCO] Source={sname} cases={len(cases)}")

    def _assign_splits_once(self) -> None:
        # target: all test
        tgt = self.target_domain
        for case_id in self._cases_by_source[tgt].keys():
            self._assigned[(tgt, case_id)] = "test"

        # non-target: all -> train pool; sample val_per_source per source for val
        for src_name in sorted(self._cases_by_source.keys()):
            if src_name == tgt:
                continue
            case_ids = sorted(list(self._cases_by_source[src_name].keys()))
            n = len(case_ids)
            if n == 0:
                continue

            k = min(self.val_per_source, n) if self.val_per_source > 0 else 0

            # deterministic per source
            seed_i = int(self.seed) ^ _stable_int_hash(src_name)
            rng = np.random.RandomState(seed_i)
            val_idx = set(rng.choice(np.arange(n), size=k, replace=False).tolist()) if k > 0 else set()

            for i, cid in enumerate(case_ids):
                self._assigned[(src_name, cid)] = "val" if i in val_idx else "train"

        # summary
        c_train = sum(1 for v in self._assigned.values() if v == "train")
        c_val = sum(1 for v in self._assigned.values() if v == "val")
        c_test = sum(1 for v in self._assigned.values() if v == "test")
        self.logger.info(
            f"[BraTS-LOCO] Split assigned (target={self.target_domain}) => "
            f"train={c_train}, val={c_val}, test={c_test}"
        )

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split_norm = self._normalize_split(split)
        if split_norm not in ("train", "val", "test"):
            raise ValueError(f"[BraTS-LOCO] split must be train/val/test, got '{split}'.")

        # Transform (project convention)
        transform = overrides.get("transform", None)
        if transform is None:
            dcfg: DictConfig = require_config(self.config, "training.data")
            tcfg: DictConfig = get_config(dcfg, "transforms", DictConfig({}))

            normalize = bool(require_config(tcfg, "normalize"))
            geom_aug = bool(require_config(tcfg, "geom_aug"))
            intensity_aug = bool(require_config(tcfg, "intensity_aug"))

            mean = get_config(tcfg, "mean", [0.0, 0.0, 0.0, 0.0])
            std = get_config(tcfg, "std", [1.0, 1.0, 1.0, 1.0])
            intensity_policy = get_config(tcfg, "intensity_policy", None)

            image_size = get_config(tcfg, "image_size", None)
            if image_size is not None:
                if len(list(image_size)) != 3:
                    raise ValueError(f"[BraTS-LOCO] training.data.transforms.image_size must be [D,H,W].")
                image_size = [int(x) for x in list(image_size)]

            transform = get_seg_transforms(
                ndim=3,
                split=split_norm,
                normalize=normalize,
                geom_aug=geom_aug,
                intensity_aug=intensity_aug,
                mean=mean,
                std=std,
                expected_label_channels=EXPECTED_REGION_CHANNELS,
                region_label_as_float=True,
                image_size=image_size,  # strict shape check only
                intensity_policy=intensity_policy,
                channel_names=list(self.modality_order),
            )

        expected_shape = overrides.get("expected_shape", self.expected_shape)
        strict_label_values = bool(overrides.get("strict_label_values", self.strict_label_values))

        # build items list
        name2src = {s.name: s for s in self.sources}
        items: List[Tuple[SourceSpec, str, Dict[str, Any]]] = []

        for (src_name, case_id), sp in self._assigned.items():
            if sp != split_norm:
                continue
            src = name2src[src_name]
            info = self._cases_by_source[src_name][case_id]
            items.append((src, case_id, info))

        if len(items) == 0:
            self.logger.warning(f"[BraTS-LOCO] split='{split_norm}' is empty (target={self.target_domain}).")
            return None

        ds = BratsDataset(
            items=items,
            split=split_norm,
            modality_order=self.modality_order,
            expected_shape=expected_shape,
            transform=transform,
            drop_unlabeled=self.drop_unlabeled,
            strict_label_values=strict_label_values,
            logger=self.logger,
        )
        return ds