"""
HECKTOR21 Multi-domain NIfTI Dataset (from a single master manifest.csv)

Design choices (aligned with your BraTS dataset style):
- Single source of truth: one manifest.csv produced by OFFLINE preprocessing.
- No online resample/resize: assumes OFFLINE preprocessing already unified shape/spacing.
- Multi-domain via "sources" filters over manifest (e.g., per center_code).
- Domain id returned to downstream is source.name (Scheme A).
- Dataset returns:
    image : FloatTensor [C, D, H, W], default C=2 for (CT, PET)
    label : FloatTensor [1, D, H, W] binary GTVt mask
- Drop unlabeled samples by default (DG/TTA evaluation requires GT).

Manifest expected columns (minimum):
  patient_id, split, status, ct_proc, pt_proc, gtvt_proc, center_code, center_id

YAML example (BraTS-like):
  name: hecktor21_nifti
  manifest_csv: /path/to/manifest.csv
  expected_shape: [144,144,144]
  drop_unlabeled: true
  patient_col: patient_id
  split_col: split
  status_col: status
  ok_status_values: ["ok"]
  ct_col: ct_proc
  pt_col: pt_proc
  label_col: gtvt_proc
  center_code_col: center_code
  center_id_col: center_id
  modality_order: ["ct","pt"]

  sources:
    - name: CHGJ
      where: { center_code: ["CHGJ"] }
      include_splits: { train: ["train"], val: ["val"], test: [] }

    - name: CHUP
      where: { center_code: ["CHUP"] }
      include_splits: { train: [], val: [], test: ["test"] }
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
# Helpers (keep consistent with BraTS)
# ----------------------------
def _load_nifti_xyz_as_canonical(path: str, dtype=np.float32) -> np.ndarray:
    """
    Load NIfTI and return array in canonical orientation (typically RAS+ for the array).
    Returned array shape is (X, Y, Z) in nibabel's array convention.
    """
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata(dtype=dtype)


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


def _to_binary_mask(y: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """
    Force label to be binary {0,1}.
    - If float, round first; if values not close to integers, warn via caller (optional).
    """
    if y.dtype.kind not in ("i", "u"):
        y_round = np.rint(y)
        # keep tolerance check but do not spam here
        y = y_round

    y = y.astype(np.int16, copy=False)
    # common cases: {0,1}, sometimes {0,255}
    uniq = np.unique(y)
    if uniq.size == 0:
        return y.astype(np.uint8, copy=False)
    if np.all(np.isin(uniq, [0, 1])):
        return y.astype(np.uint8, copy=False)
    if np.all(np.isin(uniq, [0, 255])):
        return (y // 255).astype(np.uint8, copy=False)
    # fallback: non-zero -> 1
    return (y != 0).astype(np.uint8, copy=False)


# ----------------------------
# Source spec (filter over manifest)
# ----------------------------
@dataclass(frozen=True)
class SourceFilterSpec:
    name: str
    where: Dict[str, Any]                 # e.g., {"center_code": ["CHGJ"]}
    include_splits: Dict[str, List[str]]  # per requested split -> allowed split values in manifest
    root_dir: Optional[str] = None        # optional: resolve relative paths


def _filter_manifest_for_source(
    df: pd.DataFrame,
    spec: SourceFilterSpec,
    *,
    patient_col: str,
    split_col: str,
    status_col: str,
    ok_status_values: Sequence[str],
    ct_col: str,
    pt_col: str,
    label_col: str,
    center_code_col: str,
    center_id_col: str,
    drop_unlabeled: bool,
    logger=None,
) -> pd.DataFrame:
    """
    Apply:
      - status filter
      - where filters
      - path existence sanity (optional but we keep minimal: missing path will fail later)
      - drop unlabeled (if requested)
    """
    logger = logger or get_logger()
    d = df

    # status
    ok_set = set([str(x).lower() for x in ok_status_values])
    if status_col in d.columns:
        d = d[d[status_col].astype(str).str.lower().isin(ok_set)]
    else:
        logger.warning(f"[HECKTOR21] status_col='{status_col}' not found in manifest; skip status filtering.")

    # where filters
    where = spec.where or {}
    # support keys: center_code, center_id, patient_id
    if "center_code" in where:
        vals = set([str(v).upper() for v in list(where["center_code"])])
        if center_code_col not in d.columns:
            raise ValueError(f"[HECKTOR21] center_code_col='{center_code_col}' not found in manifest.")
        d = d[d[center_code_col].astype(str).str.upper().isin(vals)]

    if "center_id" in where:
        vals = set([int(v) for v in list(where["center_id"])])
        if center_id_col not in d.columns:
            raise ValueError(f"[HECKTOR21] center_id_col='{center_id_col}' not found in manifest.")
        # numeric-safe
        d = d[pd.to_numeric(d[center_id_col], errors="coerce").fillna(-1).astype(int).isin(vals)]

    if "patient_id" in where:
        vals = set([str(v) for v in list(where["patient_id"])])
        if patient_col not in d.columns:
            raise ValueError(f"[HECKTOR21] patient_col='{patient_col}' not found in manifest.")
        d = d[d[patient_col].astype(str).isin(vals)]

    # drop unlabeled if required
    if drop_unlabeled:
        if label_col not in d.columns:
            raise ValueError(f"[HECKTOR21] label_col='{label_col}' not found in manifest.")
        d = d[d[label_col].notna() & (d[label_col].astype(str) != "")]
    return d


# ----------------------------
# Dataset
# ----------------------------
class Hecktor21Dataset(Dataset):
    """
    Multi-domain dataset built from one manifest + multiple SourceFilterSpec.

    Returns per item:
      image     : FloatTensor [2, D, H, W]  (CT, PET)
      label     : FloatTensor [1, D, H, W]  (binary GTVt)
      case_id   : str
      domain    : str  (source.name; Scheme A)
      center_code : str
      center_id : int
      index     : int
    """

    def __init__(
        self,
        manifest_csv: str,
        sources: List[SourceFilterSpec],
        split: str,
        *,
        expected_shape: Optional[Tuple[int, int, int]] = None,
        drop_unlabeled: bool = True,
        patient_col: str = "patient_id",
        split_col: str = "split",
        status_col: str = "status",
        ok_status_values: Sequence[str] = ("ok",),
        ct_col: str = "ct_proc",
        pt_col: str = "pt_proc",
        label_col: str = "gtvt_proc",
        center_code_col: str = "center_code",
        center_id_col: str = "center_id",
        modality_order: Sequence[str] = ("ct", "pt"),
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
        logger=None,
        strict_label_values: bool = True,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.split = str(split).lower()
        self.expected_shape = expected_shape
        self.drop_unlabeled = bool(drop_unlabeled)
        self.transform = transform
        self.strict_label_values = bool(strict_label_values)

        self.patient_col = patient_col
        self.split_col = split_col
        self.status_col = status_col
        self.ok_status_values = [str(x).lower() for x in list(ok_status_values)]
        self.ct_col = ct_col
        self.pt_col = pt_col
        self.label_col = label_col
        self.center_code_col = center_code_col
        self.center_id_col = center_id_col

        self.modality_order = [m.lower() for m in modality_order]
        if self.modality_order != ["ct", "pt"]:
            self.logger.warning(
                f"[HECKTOR21] modality_order={self.modality_order}. Expected ['ct','pt'] in this project."
            )

        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"[HECKTOR21] manifest_csv not found: {manifest_csv}")

        df = pd.read_csv(manifest_csv)

        # required cols check (minimal)
        for c in [patient_col, split_col, ct_col, pt_col]:
            if c not in df.columns:
                raise ValueError(f"[HECKTOR21] manifest missing required column '{c}': {manifest_csv}")

        if label_col not in df.columns:
            df[label_col] = np.nan

        if center_code_col not in df.columns:
            df[center_code_col] = ""
        if center_id_col not in df.columns:
            df[center_id_col] = np.nan

        # build index like BraTS: List[(src_spec, case_id, row_dict)]
        self._index: List[Tuple[SourceFilterSpec, str, Dict[str, Any]]] = []

        for src in sources:
            dsrc = _filter_manifest_for_source(
                df=df,
                spec=src,
                patient_col=patient_col,
                split_col=split_col,
                status_col=status_col,
                ok_status_values=self.ok_status_values,
                ct_col=ct_col,
                pt_col=pt_col,
                label_col=label_col,
                center_code_col=center_code_col,
                center_id_col=center_id_col,
                drop_unlabeled=self.drop_unlabeled,
                logger=self.logger,
            )

            include_vals = src.include_splits.get(self.split, [self.split])
            include_vals = [str(v).lower() for v in list(include_vals)]

            # filter by split values in manifest
            dsrc = dsrc[dsrc[split_col].astype(str).str.lower().isin(include_vals)]

            for _, row in dsrc.iterrows():
                case_id = str(row[patient_col])
                info = row.to_dict()
                self._index.append((src, case_id, info))

        if len(self._index) == 0:
            raise ValueError(
                f"[HECKTOR21] No samples after filtering. split='{self.split}'. "
                f"Check include_splits and manifest 'split' values."
            )

        self.logger.info(
            f"[HECKTOR21] Built dataset: split='{self.split}', n={len(self._index)}, "
            f"sources={[s.name for s in sources]}"
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src, case_id, info = self._index[idx]

        # paths
        ct_path = _resolve_path(info.get(self.ct_col, ""), src.root_dir)
        pt_path = _resolve_path(info.get(self.pt_col, ""), src.root_dir)
        lb_path = _resolve_path(info.get(self.label_col, ""), src.root_dir)

        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"[HECKTOR21] Missing CT file: {ct_path} (case={case_id}, domain={src.name})")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"[HECKTOR21] Missing PET file: {pt_path} (case={case_id}, domain={src.name})")

        if self.drop_unlabeled:
            if (not lb_path) or (not os.path.exists(lb_path)):
                raise FileNotFoundError(
                    f"[HECKTOR21] Missing label file: {lb_path} (case={case_id}, domain={src.name})"
                )

        # ---- load (X,Y,Z) canonical ----
        ct = _load_nifti_xyz_as_canonical(ct_path, dtype=np.float32)
        pt = _load_nifti_xyz_as_canonical(pt_path, dtype=np.float32)
        _validate_shape(ct, self.expected_shape, what="ct", case_id=case_id)
        _validate_shape(pt, self.expected_shape, what="pt", case_id=case_id)

        # stack -> [C,X,Y,Z]
        image = np.stack([ct, pt], axis=0).astype(np.float32, copy=False)
        # to torch -> [C,D,H,W] where D=Z, H=Y, W=X (same as BraTS)
        image_t = torch.from_numpy(image).float().permute(0, 3, 2, 1)

        # ---- label ----
        if (not lb_path) or (not os.path.exists(lb_path)):
            # safe fallback (should not happen with drop_unlabeled=True)
            y_np = np.zeros(ct.shape, dtype=np.uint8)
        else:
            y = _load_nifti_xyz_as_canonical(lb_path, dtype=np.float32)
            _validate_shape(y, self.expected_shape, what="label", case_id=case_id)
            y_np = _to_binary_mask(y)

        # label -> torch [1,D,H,W] float
        y_t = torch.from_numpy(y_np.astype(np.float32, copy=False)).permute(2, 1, 0).unsqueeze(0)

        if self.strict_label_values:
            uniq = torch.unique(y_t).detach().cpu().tolist()
            bad = [v for v in uniq if v not in (0.0, 1.0)]
            if len(bad) > 0:
                raise ValueError(
                    f"[HECKTOR21] Label must be binary {{0,1}}. got values={uniq} "
                    f"(case={case_id}, domain={src.name})."
                )

        # ---- transform ----
        if self.transform is not None:
            out = self.transform(image_t, y_t)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                image_t, y_t = out
            else:
                raise RuntimeError(
                    "[HECKTOR21] transform must return (image, label), "
                    f"got type={type(out)}"
                )

        # final guard: label channel = 1
        if y_t.ndim != 4 or int(y_t.shape[0]) != 1:
            raise ValueError(
                f"[HECKTOR21] Label shape must be [1,D,H,W], got {tuple(y_t.shape)} "
                f"(case={case_id}, domain={src.name})"
            )

        center_code = str(info.get(self.center_code_col, ""))
        center_id = info.get(self.center_id_col, None)
        try:
            center_id = int(center_id) if (center_id is not None and str(center_id) != "nan") else -1
        except Exception:
            center_id = -1

        return {
            "image": image_t,           # [2,D,H,W]
            "label": y_t,               # [1,D,H,W]
            "case_id": case_id,
            "domain": src.name,         # Scheme A
            "center_code": center_code,
            "center_id": center_id,
            "split": str(info.get(self.split_col, "")).lower(),
            "index": int(idx),
        }


# ----------------------------
# Builder / Registry
# ----------------------------
@register_dataset_builder("hecktor21")
class Hecktor21Builder(BaseDatasetBuilder):
    """
    Config schema (BraTS-like):

    dataset:
      name: hecktor21_nifti
      manifest_csv: /path/to/manifest.csv

      expected_shape: [144,144,144]
      drop_unlabeled: true
      strict_label_values: true
      modality_order: ["ct","pt"]

      patient_col: patient_id
      split_col: split
      status_col: status
      ok_status_values: ["ok"]

      ct_col: ct_proc
      pt_col: pt_proc
      label_col: gtvt_proc

      center_code_col: center_code
      center_id_col: center_id

      sources:
        - name: CHGJ
          where:
            center_code: ["CHGJ"]
          include_splits:
            train: ["train"]
            val:   ["val"]
            test:  []

        - name: CHUP
          where:
            center_code: ["CHUP"]
          include_splits:
            train: []
            val:   []
            test:  ["test"]
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
        self.split_col = str(get_config(dcfg, "split_col", "split"))
        self.status_col = str(get_config(dcfg, "status_col", "status"))
        self.ok_status_values = list(get_config(dcfg, "ok_status_values", ["ok"]))

        self.ct_col = str(get_config(dcfg, "ct_col", "ct_proc"))
        self.pt_col = str(get_config(dcfg, "pt_col", "pt_proc"))
        self.label_col = str(get_config(dcfg, "label_col", "gtvt_proc"))

        self.center_code_col = str(get_config(dcfg, "center_code_col", "center_code"))
        self.center_id_col = str(get_config(dcfg, "center_id_col", "center_id"))

        self.modality_order = tuple([str(x).lower() for x in list(get_config(dcfg, "modality_order", ["ct", "pt"]))])

        sources_cfg = get_config(dcfg, "sources", None)
        if sources_cfg is None:
            raise ValueError("[hecktor21] 'dataset.sources' is required for multi-domain loading.")

        self.sources: List[SourceFilterSpec] = []
        for sc in sources_cfg:
            sname = str(require_config(sc, "name", type_=str))
            where = dict(get_config(sc, "where", DictConfig({})))
            include_splits = get_config(sc, "include_splits", DictConfig({}))
            include_splits = {
                k.lower(): [str(v).lower() for v in list(vals)]
                for k, vals in dict(include_splits).items()
            }
            include_splits.setdefault("train", ["train"])
            include_splits.setdefault("val", ["val"])
            include_splits.setdefault("test", ["test"])

            root_dir = get_config(sc, "root_dir", None)

            self.sources.append(
                SourceFilterSpec(
                    name=sname,
                    where=where,
                    include_splits=include_splits,
                    root_dir=root_dir,
                )
            )

    def build_dataset(self, split: str, **overrides) -> Optional[Dataset]:
        split_norm = self._normalize_split(split)

        # short-circuit: if this split is disabled by config for ALL sources -> return None
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
                f"[hecktor21] split='{split_norm}' is disabled by include_splits for all sources; return None."
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

            # mean/std for 2 channels; provide defaults if not set
            mean = get_config(tcfg, "mean", [0.0, 0.0])
            std = get_config(tcfg, "std", [1.0, 1.0])

            intensity_policy = get_config(tcfg, "intensity_policy", None)

            # shape check only (no resize)
            image_size = get_config(tcfg, "image_size", None)
            if image_size is not None:
                if len(list(image_size)) != 3:
                    raise ValueError(
                        f"[hecktor21] training.data.transforms.image_size must be [D,H,W], got {list(image_size)}"
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
                expected_label_channels=1,
                region_label_as_float=True,  # label is mask float
                image_size=image_size,       # strict shape check only
                intensity_policy=intensity_policy,
                channel_names=list(self.modality_order), # ["ct", "pt"]
            )

        expected_shape = overrides.get("expected_shape", self.expected_shape)
        strict_label_values = bool(overrides.get("strict_label_values", self.strict_label_values))

        ds = Hecktor21Dataset(
            manifest_csv=self.manifest_csv,
            sources=self.sources,
            split=split_norm,
            expected_shape=expected_shape,
            drop_unlabeled=self.drop_unlabeled,
            patient_col=self.patient_col,
            split_col=self.split_col,
            status_col=self.status_col,
            ok_status_values=self.ok_status_values,
            ct_col=self.ct_col,
            pt_col=self.pt_col,
            label_col=self.label_col,
            center_code_col=self.center_code_col,
            center_id_col=self.center_id_col,
            modality_order=self.modality_order,
            transform=transform,
            logger=self.logger,
            strict_label_values=strict_label_values,
        )
        return ds