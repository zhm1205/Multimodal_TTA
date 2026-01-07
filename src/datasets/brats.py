# file: src/datasets/brats_multi_nifti.py
"""
BraTS24 (GLI / SSA / PED) Multi-source NIfTI Dataset

Design choices (per your latest requirements):
- Multi-source via multiple processed.csv files (GLI / SSA / PED) with an explicit `split` column.
- No online resize / resample: assumes OFFLINE preprocessing already unified spacing/shape.
- Always use background=0 data (img_col defaults to "img_path"; no QC variant).
- Drop unlabeled samples (DG/TTA evaluation requires GT).
- Dataset returns ONLY the final supervision signal used for training:
    label : region masks [R, D, H, W] (FloatTensor), default R=3 for ET/TC/WT.
  Raw label id map is loaded only to build regions, then discarded.

CSV expected columns (minimum):
  subject_id, modality, img_path, label_path, split

Modalities expected (default):
  t1n, t1c, t2w, t2f

Profiles / raw label semantics (as described in your header):
  - gli/ssa: NETC=1, SNFH=2, ET=3, (gli only) RC=4
      WT = {1,2,3}, TC = {1,3}, ET = {3}
    Note: RC=4 is ignored by default in WT/TC/ET (you can override region_map if needed).
  - ped: ET=1, NET=2, CC=3, ED=4
      WT = {1,2,3,4}, TC = {1,2,3}, ET = {1}

Domain identification:
- Each sample includes "domain" (source name) + "profile", so downstream can separate test domains.
"""

from __future__ import annotations

import os
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
# Hard-coded BraTS region definition (what you train/evaluate)
# ----------------------------
REGION_ORDER: List[str] = ["ET", "TC", "WT"]
EXPECTED_REGION_CHANNELS: int = 3  # write it in code as you requested


DEFAULT_REGION_MAPS: Dict[str, Dict[str, List[int]]] = {
    # GLI / SSA share the same taxonomy for NETC(1), SNFH(2), ET(3); GLI may also include RC(4).
    # Default BraTS regions ignore RC=4 unless you override.
    "gli": {
        "ET": [3],
        "TC": [1, 3],
        "WT": [1, 2, 3],
    },
    "ssa": {
        "ET": [3],
        "TC": [1, 3],
        "WT": [1, 2, 3],
    },
    # PED taxonomy: ET(1), NET(2), CC(3), ED(4)
    "ped": {
        "ET": [1],
        "TC": [1, 2, 3],
        "WT": [1, 2, 3, 4],
    },
}


# ----------------------------
# Helpers
# ----------------------------

def _load_nifti_xyz_as_canonical(path: str, dtype=np.float32) -> np.ndarray:
    """
    Load NIfTI and return array in canonical orientation.
    nibabel.as_closest_canonical typically yields RAS+ orientation for the array.
    Returned array shape is (X, Y, Z) in nibabel's array convention.
    """
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(dtype=dtype)


def _safe_round_label(label: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """
    Ensure label becomes integer-valued.

    If you see non-integer values here (max_err > tol), that usually means labels were
    resampled with non-nearest interpolation somewhere in preprocessing.
    We still round to recover, but you should fix the pipeline.
    """
    if label.dtype.kind in ("i", "u"):
        return label.astype(np.int16, copy=False)

    rounded = np.rint(label)
    if label.size > 0:
        max_err = float(np.max(np.abs(label - rounded)))
        # Do not spam logs here; caller can choose strict checks.
        _ = max_err > tol
    return rounded.astype(np.int16, copy=False)


def _validate_shape(arr: np.ndarray, expected_shape: Optional[Tuple[int, int, int]], what: str, case_id: str) -> None:
    if expected_shape is None:
        return
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"[BraTS-Multi-NIfTI] Shape mismatch for {what} case={case_id}: "
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
    """
    Build region masks in fixed REGION_ORDER (ET,TC,WT).
    Returns: [R,D,H,W] float32
    """
    masks: List[torch.Tensor] = []
    for rname in REGION_ORDER:
        vals = region_map.get(rname, [])
        if len(vals) == 0:
            masks.append(torch.zeros_like(y_id, dtype=torch.float32))
            continue
        m = torch.zeros_like(y_id, dtype=torch.bool)
        for v in vals:
            m |= (y_id == int(v))
        masks.append(m.float())
    y_reg = torch.stack(masks, dim=0)  # [R,D,H,W]
    return y_reg


# ----------------------------
# CSV Parsing
# ----------------------------

@dataclass(frozen=True)
class SourceSpec:
    name: str
    csv_path: str
    profile: str
    root_dir: Optional[str]
    include_splits: Dict[str, List[str]]  # per requested split -> allowed split values in csv
    region_map: Dict[str, List[int]]      # per profile (can override)


def _parse_processed_csv_to_cases(
    csv_path: str,
    modality_order: Sequence[str],
    *,
    root_dir: Optional[str],
    drop_unlabeled: bool,
    # columns are fixed to keep this simple
    split_col: str = "split",
    subject_col: str = "subject_id",
    modality_col: str = "modality",
    img_col: str = "img_path",
    label_col: str = "label_path",
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return:
      cases[case_id] = {
         "split": <split_str>,
         "modalities": {mod: img_path},
         "label": label_path
      }
    """
    logger = logger or get_logger()
    df = pd.read_csv(csv_path)

    for c in [subject_col, modality_col, img_col, split_col]:
        if c not in df.columns:
            raise ValueError(f"[BraTS-Multi-NIfTI] CSV missing required column '{c}': {csv_path}")

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
            # split consistency check
            if cases[case_id]["split"] != split:
                logger.warning(
                    f"[BraTS-Multi-NIfTI] Inconsistent split for case={case_id}: "
                    f"{cases[case_id]['split']} vs {split}. Keep the first."
                )
            # label consistency check
            if label_path and cases[case_id]["label"] and (cases[case_id]["label"] != label_path):
                logger.warning(
                    f"[BraTS-Multi-NIfTI] Inconsistent label_path for case={case_id}: "
                    f"{cases[case_id]['label']} vs {label_path}. Keep the first."
                )
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
        f"[BraTS-Multi-NIfTI] Parsed {csv_path}: total_cases={len(cases)}, "
        f"valid_cases={len(valid)}, dropped_missing_mod={dropped_missing_mod}, "
        f"dropped_no_label={dropped_no_label}"
    )
    return valid


# ----------------------------
# Dataset
# ----------------------------

class BratsMultiSourceNiftiDataset(Dataset):
    """
    Multi-source dataset from multiple CSV sources.

    Returns per item:
      image  : FloatTensor [C,D,H,W]
      label  : FloatTensor [3,D,H,W]  (ET/TC/WT in fixed order)
      case_id: str
      domain : str (source name)
      profile: str
      index  : int
    """

    def __init__(
        self,
        sources: List[SourceSpec],
        split: str,
        modality_order: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
        expected_shape: Optional[Tuple[int, int, int]] = None,
        drop_unlabeled: bool = True,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
        strict_label_values: bool = False,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.split = str(split).lower()
        self.modality_order = [m.lower() for m in modality_order]
        self.expected_shape = expected_shape
        self.drop_unlabeled = bool(drop_unlabeled)
        self.transform = transform
        self.strict_label_values = bool(strict_label_values)

        self._index: List[Tuple[SourceSpec, str, Dict[str, Any]]] = []

        for src in sources:
            if not os.path.exists(src.csv_path):
                raise FileNotFoundError(f"[BraTS-Multi-NIfTI] CSV not found: {src.csv_path}")

            cases = _parse_processed_csv_to_cases(
                csv_path=src.csv_path,
                modality_order=self.modality_order,
                root_dir=src.root_dir,
                drop_unlabeled=self.drop_unlabeled,
                logger=self.logger,
            )

            include_vals = src.include_splits.get(self.split, [self.split])
            include_vals = [str(v).lower() for v in include_vals]

            for case_id, info in cases.items():
                if str(info["split"]).lower() in include_vals:
                    self._index.append((src, case_id, info))

        if len(self._index) == 0:
            raise ValueError(
                f"[BraTS-Multi-NIfTI] No samples after filtering. split='{self.split}'. "
                f"Check include_splits and CSV 'split' values."
            )

        self.logger.info(
            f"[BraTS-Multi-NIfTI] Built dataset: split='{self.split}', n={len(self._index)}, "
            f"sources={[s.name for s in sources]}"
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src, case_id, info = self._index[idx]

        # ---- load image modalities (assume offline preprocessed and aligned) ----
        img_list: List[np.ndarray] = []
        for mod in self.modality_order:
            p = info["modalities"][mod]
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[BraTS-Multi-NIfTI] Missing image file: {p} (case={case_id}, mod={mod})"
                )
            vol = _load_nifti_xyz_as_canonical(p, dtype=np.float32)  # [X,Y,Z]
            _validate_shape(vol, self.expected_shape, what=f"image/{mod}", case_id=case_id)
            img_list.append(vol)

        # Stack: [C, X, Y, Z]
        image = np.stack(img_list, axis=0).astype(np.float32, copy=False)

        # Convert to torch: [C,Z,Y,X]  => spatial = (Z,Y,X) = (160,196,160)
        image_t = torch.from_numpy(image).float().permute(0, 3, 2, 1)  # [C,Z,Y,X]

        # ---- load raw label id map, then build region masks ----
        label_path = info.get("label", "")
        if (not label_path) or (not os.path.exists(label_path)):
            # should not happen with drop_unlabeled=True; keep safe
            y_np = np.zeros(image.shape[1:], dtype=np.int16)
        else:
            y = _load_nifti_xyz_as_canonical(label_path, dtype=np.float32)  # maybe float if pipeline was wrong
            _validate_shape(y, self.expected_shape, what="label", case_id=case_id)
            y_np = _safe_round_label(y)  # int16

        # raw id to torch [D,H,W] = [Z,Y,X]
        y_id = torch.from_numpy(y_np.astype(np.int64, copy=False)).long().permute(2, 1, 0)  # [Z,Y,X]

        # Optional strict sanity check: reject weird values early
        if self.strict_label_values:
            uniq = torch.unique(y_id).detach().cpu().tolist()
            # just guard: BraTS labels should be small non-negative ints
            bad = [v for v in uniq if (v < 0 or v > 20)]
            if len(bad) > 0:
                raise ValueError(
                    f"[BraTS-Multi-NIfTI] Abnormal label values {bad} in case={case_id} src={src.name}. "
                    f"This often indicates non-nearest interpolation in preprocessing."
                )

        # Build region masks [3,D,H,W] float32 (ET/TC/WT)
        y_reg = _build_region_masks_from_raw(y_id, region_map=src.region_map)

        # ---- transform: apply to (image, region label) only ----
        if self.transform is not None:
            out = self.transform(image_t, y_reg)  # label is [3,D,H,W]
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image_t, y_reg = out
            else:
                raise RuntimeError(
                    "[BraTS-Multi-NIfTI] transform must return (image, label_region), "
                    f"got type={type(out)}"
                )

        # Final guard: ensure label is [3,D,H,W]
        if y_reg.ndim != 4 or int(y_reg.shape[0]) != EXPECTED_REGION_CHANNELS:
            raise ValueError(
                f"[BraTS-Multi-NIfTI] Region label shape must be [3,D,H,W], got {tuple(y_reg.shape)} "
                f"(case={case_id}, src={src.name})"
            )

        return {
            "image": image_t,   # [C,D,H,W]
            "label": y_reg,     # [3,D,H,W] (ET/TC/WT)
            "case_id": case_id,
            "domain": src.name,
            "profile": src.profile,
            "index": int(idx),
        }


# ----------------------------
# Builder / Registry
# ----------------------------

@register_dataset_builder("brats")
class BratsMultiNiftiBuilder(BaseDatasetBuilder):
    """
    Minimal config (kept intentionally simple):

    dataset:
      name: brats_multi_nifti
      expected_shape: [160,196,160]     # recommended (offline-preprocessed)
      strict_label_values: false

      sources:
        - name: brats24_gli
          profile: gli
          csv_path: /path/BraTS24-GLI/processed.csv
          root_dir: null
          include_splits:
            train: ["train"]
            val:   ["val"]
            test:  ["test"]

        - name: brats24_ssa
          profile: ssa
          csv_path: /path/BraTS24-SSA/processed.csv
          include_splits:
            train: []
            val:   []
            test:  ["train","val","test"]

        - name: brats24_ped
          profile: ped
          csv_path: /path/BraTS24-PED/processed.csv
          include_splits:
            train: []
            val:   []
            test:  ["train","val","test"]
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        dcfg: DictConfig = require_config(config, "dataset")

        exp_shape = get_config(dcfg, "expected_shape", None)
        self.expected_shape = tuple(exp_shape) if exp_shape is not None else None

        self.strict_label_values = bool(get_config(dcfg, "strict_label_values", False))

        # Sources are required (multi-source is the point here)
        sources_cfg = get_config(dcfg, "sources", None)
        if sources_cfg is None:
            raise ValueError("[brats_multi_nifti] 'dataset.sources' is required for multi-source loading.")

        self.sources: List[SourceSpec] = []
        for sc in sources_cfg:
            sname = str(require_config(sc, "name", type_=str))
            csv_path = str(require_config(sc, "csv_path", type_=str))
            profile = str(get_config(sc, "profile", "gli")).lower()
            root_dir = get_config(sc, "root_dir", None)

            include_splits = get_config(sc, "include_splits", DictConfig({}))
            include_splits = {
                k.lower(): [str(v).lower() for v in list(vals)]
                for k, vals in dict(include_splits).items()
            }
            # Default if omitted
            include_splits.setdefault("train", ["train"])
            include_splits.setdefault("val", ["val"])
            include_splits.setdefault("test", ["test"])

            # Region map: profile default; allow override but not required
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
                    include_splits=include_splits,
                    region_map=region_map,
                )
            )

        # modality_order is fixed for BraTS in this project (hard-coded as per your request)
        self.modality_order = ("t1n", "t1c", "t2w", "t2f")

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split_norm = self._normalize_split(split)

        # ----------------------------
        # Short-circuit: if this split is disabled by config for ALL sources,
        # do NOT construct dataset (avoid ValueError in dataset __init__).
        # This is required when you set include_splits.val: [] (skip validation).
        # ----------------------------
        enabled = False
        for s in self.sources:
            vals = s.include_splits.get(split_norm, None)
            if vals is None:
                continue
            if len(list(vals)) > 0:
                enabled = True
                break

        if not enabled:
            self.logger.warning(
                f"[brats_multi_nifti] split='{split_norm}' is disabled by include_splits for all sources; "
                f"return None."
            )
            return None

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

            # Use existing config field for shape check ONLY (no resize/crop/pad)
            image_size = get_config(tcfg, "image_size", None)
            if image_size is not None:
                # must be [D,H,W]
                if len(list(image_size)) != 3:
                    raise ValueError(
                        f"[brats_multi_nifti] training.data.transforms.image_size must be [D,H,W], "
                        f"got {list(image_size)}"
                    )
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
                image_size=image_size,   # <-- enables strict shape check in transforms.py
            )

        expected_shape = overrides.get("expected_shape", self.expected_shape)
        strict_label_values = bool(overrides.get("strict_label_values", self.strict_label_values))

        ds = BratsMultiSourceNiftiDataset(
            sources=self.sources,
            split=split_norm,
            modality_order=self.modality_order,
            expected_shape=expected_shape,
            drop_unlabeled=True,
            transform=transform,
            logger=self.logger,
            strict_label_values=strict_label_values,
        )
        return ds