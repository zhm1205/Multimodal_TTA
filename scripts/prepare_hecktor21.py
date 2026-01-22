# prep_hecktor21.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import yaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Please `pip install pyyaml`.") from e

try:
    import SimpleITK as sitk
except Exception as e:
    raise RuntimeError(
        "Missing dependency: SimpleITK. Please `pip install SimpleITK`.\n"
        "SimpleITK is strongly recommended for robust physical-space bbox cropping."
    ) from e


# -----------------------------
# Helpers
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def patient_center_code(patient_id: str) -> str:
    m = re.match(r"^([A-Za-z]{4})", patient_id)
    if not m:
        return "UNK"
    return m.group(1).upper()


def get_interpolator(name: str) -> int:
    name = name.lower().strip()
    if name in ["linear", "lin"]:
        return sitk.sitkLinear
    if name in ["nearest", "nn", "nearestneighbor"]:
        return sitk.sitkNearestNeighbor
    if name in ["bspline", "b-spline"]:
        return sitk.sitkBSpline
    raise ValueError(f"Unknown interpolator: {name}")


def cast_float_dtype(img: sitk.Image, dtype: str) -> sitk.Image:
    dtype = dtype.lower().strip()
    if dtype == "float32":
        return sitk.Cast(img, sitk.sitkFloat32)
    if dtype == "float16":
        return sitk.Cast(img, sitk.sitkFloat16)
    if dtype == "float64":
        return sitk.Cast(img, sitk.sitkFloat64)
    raise ValueError(f"Unsupported float dtype: {dtype}")


def cast_mask_dtype(img: sitk.Image, dtype: str) -> sitk.Image:
    dtype = dtype.lower().strip()
    if dtype in ["uint8", "u8"]:
        return sitk.Cast(img, sitk.sitkUInt8)
    if dtype in ["uint16", "u16"]:
        return sitk.Cast(img, sitk.sitkUInt16)
    if dtype in ["int16", "i16"]:
        return sitk.Cast(img, sitk.sitkInt16)
    raise ValueError(f"Unsupported mask dtype: {dtype}")


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float,
    out_pixel_type: int,
) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),  # identity
        interpolator,
        default_value,
        out_pixel_type,
    )


def resample_to_spacing(
    img: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolator: int,
    default_value: float,
    out_pixel_type: int,
) -> sitk.Image:
    """
    Resample `img` to target_spacing while preserving physical space (origin/direction) and FOV.
    New size is computed from old_size * old_spacing / new_spacing (rounded).
    """
    old_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)
    old_size = np.array(list(img.GetSize()), dtype=np.int64)

    new_spacing = np.array(list(target_spacing), dtype=np.float64)
    new_size = np.round(old_size * (old_spacing / new_spacing)).astype(np.int64)
    new_size = np.maximum(new_size, 1).tolist()

    ref = sitk.Image([int(x) for x in new_size], out_pixel_type)
    ref.SetSpacing([float(x) for x in new_spacing.tolist()])
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(img.GetOrigin())

    out = sitk.Resample(img, ref, sitk.Transform(), interpolator, default_value, out_pixel_type)
    return out


def bbox_mm_to_index_roi(
    img: sitk.Image,
    x1: float, x2: float,
    y1: float, y2: float,
    z1: float, z2: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Convert a physical-space bbox (mm, ITK convention) into an axis-aligned index ROI on img grid.
    Robust to direction flips by converting all 8 corners and taking min/max.
    Returns:
      - start_index [i,j,k]
      - roi_size [si,sj,sk]
      - debug dict
    """
    xs = [x1, x2]
    ys = [y1, y2]
    zs = [z1, z2]
    corners = [(x, y, z) for x in xs for y in ys for z in zs]

    idxs = []
    for p in corners:
        ci = img.TransformPhysicalPointToContinuousIndex(p)
        idxs.append(ci)

    idxs = np.array(idxs, dtype=np.float64)  # (8, 3)
    mins = idxs.min(axis=0)
    maxs = idxs.max(axis=0)

    start = np.floor(mins).astype(int)
    end = np.ceil(maxs).astype(int)

    size = (end - start + 1).astype(int)

    dbg = {
        "corners_mm": corners,
        "corners_cont_idx": idxs.tolist(),
        "mins_cont_idx": mins.tolist(),
        "maxs_cont_idx": maxs.tolist(),
        "start_idx": start.tolist(),
        "end_idx": end.tolist(),
        "roi_size": size.tolist(),
    }
    return start.tolist(), size.tolist(), dbg


def pad_if_needed(
    img: sitk.Image,
    start_idx: List[int],
    roi_size: List[int],
    pad_value: float,
) -> Tuple[sitk.Image, List[int], Dict[str, Any]]:
    img_size = np.array(list(img.GetSize()), dtype=int)
    start = np.array(start_idx, dtype=int)
    size = np.array(roi_size, dtype=int)
    end = start + size - 1  # inclusive

    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(end - (img_size - 1), 0)

    pad_before_l = pad_before.tolist()
    pad_after_l = pad_after.tolist()

    if np.any(pad_before > 0) or np.any(pad_after > 0):
        padded = sitk.ConstantPad(img, pad_before_l, pad_after_l, pad_value)
        new_start = (start + pad_before).tolist()
        dbg = {
            "padded": True,
            "pad_before": pad_before_l,
            "pad_after": pad_after_l,
            "orig_size": img_size.tolist(),
            "new_size": list(padded.GetSize()),
        }
        return padded, new_start, dbg

    dbg = {
        "padded": False,
        "pad_before": [0, 0, 0],
        "pad_after": [0, 0, 0],
        "orig_size": img_size.tolist(),
        "new_size": img_size.tolist(),
    }
    return img, start_idx, dbg


def crop_roi(img: sitk.Image, start_idx: List[int], roi_size: List[int]) -> sitk.Image:
    return sitk.RegionOfInterest(img, roi_size, start_idx)


def compute_center_pad_crop_params(
    cur_size: List[int],
    target_size: List[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Compute center-aligned pad/crop params to transform cur_size -> target_size.
    Returns:
      pad_before, pad_after, crop_lower, crop_upper  (all length 3)
    Apply: crop then pad.
    """
    cur = np.array(cur_size, dtype=int)
    tgt = np.array(target_size, dtype=int)
    diff = tgt - cur  # positive => need pad; negative => need crop

    pad_before = np.zeros(3, dtype=int)
    pad_after = np.zeros(3, dtype=int)
    crop_lower = np.zeros(3, dtype=int)
    crop_upper = np.zeros(3, dtype=int)

    for d in range(3):
        if diff[d] >= 0:
            pb = diff[d] // 2
            pa = diff[d] - pb
            pad_before[d] = pb
            pad_after[d] = pa
        else:
            cut = -diff[d]
            cl = cut // 2
            cu = cut - cl
            crop_lower[d] = cl
            crop_upper[d] = cu

    return pad_before.tolist(), pad_after.tolist(), crop_lower.tolist(), crop_upper.tolist()


def apply_center_pad_crop(
    img: sitk.Image,
    target_size: List[int],
    pad_value: float,
    pad_before: List[int],
    pad_after: List[int],
    crop_lower: List[int],
    crop_upper: List[int],
) -> sitk.Image:
    """
    Apply center crop (if needed) then constant pad (if needed) to reach target_size.
    """
    if any(v > 0 for v in crop_lower) or any(v > 0 for v in crop_upper):
        img = sitk.Crop(img, crop_lower, crop_upper)

    if any(v > 0 for v in pad_before) or any(v > 0 for v in pad_after):
        img = sitk.ConstantPad(img, pad_before, pad_after, pad_value)

    if list(img.GetSize()) != [int(x) for x in target_size]:
        raise RuntimeError(
            f"[pad/crop] failed to reach target_size={target_size}, got={list(img.GetSize())}"
        )
    return img


# -----------------------------
# Split logic
# -----------------------------
def assign_splits(
    df: pd.DataFrame,
    enable_split: bool,
    source_centers: List[str],
    target_centers: List[str],
    val_per_center: int,
    seed: int,
    other_policy: str,
) -> pd.DataFrame:
    """
    Adds columns: domain, split.
    - If enable_split=False: domain=all, split=train for all rows (debug mode).
    - If enable_split=True:
        * target centers -> domain=target, split=test (ALL)
        * source centers -> domain=source, split=train; then sample val_per_center per center => split=val
        * others -> domain per other_policy (source/target/ignore), then split accordingly
    """
    df = df.copy()

    if not enable_split:
        df["domain"] = "all"
        df["split"] = "train"
        return df

    source_centers = [str(c).upper() for c in source_centers]
    target_centers = [str(c).upper() for c in target_centers]
    other_policy = str(other_policy).lower().strip()

    def domain_from_center(c: str) -> str:
        c = str(c).upper()
        if c in source_centers:
            return "source"
        if c in target_centers:
            return "target"
        if other_policy == "source":
            return "source"
        if other_policy == "target":
            return "target"
        return "ignore"

    df["domain"] = df["center_code"].map(domain_from_center)

    df["split"] = "ignore"
    df.loc[df["domain"] == "target", "split"] = "test"
    df.loc[df["domain"] == "source", "split"] = "train"

    rng = np.random.RandomState(seed)
    for center in sorted(set(df.loc[df["domain"] == "source", "center_code"].tolist())):
        idxs = df.index[(df["domain"] == "source") & (df["center_code"] == center)].tolist()
        if len(idxs) == 0:
            continue
        k = min(val_per_center, len(idxs))
        val_idxs = rng.choice(idxs, size=k, replace=False).tolist()
        df.loc[val_idxs, "split"] = "val"

    return df


# -----------------------------
# CSV-only manifest builder
# -----------------------------
def build_manifest_csv_only(
    df: pd.DataFrame,
    nii_root: Path,
    out_root: Path,
    out_manifest_csv: Path,
    export_per_domain_csv: bool,
    ct_suffix: str,
    pt_suffix: str,
    gt_suffix: str,
) -> pd.DataFrame:
    """
    Only writes a manifest CSV. Does NOT read or write images.
    It still fills *_raw and expected *_proc paths for downstream convenience.
    """
    img_out_dir = out_root / "images"
    lab_out_dir = out_root / "labels"
    ensure_dir(img_out_dir)
    ensure_dir(lab_out_dir)
    ensure_dir(out_manifest_csv.parent)

    rows = []
    for _, r in df.iterrows():
        pid = str(r["PatientID"])
        center_code = str(r["center_code"])
        center_id = r.get("CenterID", None)
        domain = str(r.get("domain", ""))
        split = str(r.get("split", ""))

        if split == "ignore" or domain == "ignore":
            continue

        ct_path = nii_root / f"{pid}{ct_suffix}"
        pt_path = nii_root / f"{pid}{pt_suffix}"
        gt_path = nii_root / f"{pid}{gt_suffix}"

        ct_out = img_out_dir / f"{pid}_ct.nii.gz"
        pt_out = img_out_dir / f"{pid}_pt.nii.gz"
        gt_out = lab_out_dir / f"{pid}_gtvt.nii.gz"

        status = "ok" if (ct_path.exists() and pt_path.exists() and gt_path.exists()) else "missing_file"

        rows.append({
            "patient_id": pid,
            "center_code": center_code,
            "center_id": center_id,
            "domain": domain,
            "split": split,
            "status": status,

            "ct_raw": str(ct_path),
            "pt_raw": str(pt_path),
            "gtvt_raw": str(gt_path),

            # expected processed paths (even if not yet generated)
            "ct_proc": str(ct_out),
            "pt_proc": str(pt_out),
            "gtvt_proc": str(gt_out),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_manifest_csv, index=False)

    if export_per_domain_csv and len(df_out) > 0:
        src = df_out[df_out["domain"] == "source"].copy()
        tgt = df_out[df_out["domain"] == "target"].copy()
        if len(src) > 0:
            src.to_csv(out_manifest_csv.with_name("source.csv"), index=False)
        if len(tgt) > 0:
            tgt.to_csv(out_manifest_csv.with_name("target.csv"), index=False)

    return df_out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--mode",
        choices=["full", "split_only"],
        default="full",
        help="full: preprocess images and write nii.gz; split_only: only write/update manifest CSV (no IO for images).",
    )
    ap.add_argument("--workers", type=int, default=1, help="Reserved for future; currently single-process.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    bbox_csv = cfg["bbox_csv"]
    info_csv = cfg["info_csv"]
    nii_root = Path(cfg["nii_root"])

    out_root = Path(cfg["out_root"])
    out_manifest_csv = Path(cfg["out_manifest_csv"])
    export_per_domain_csv = bool(cfg.get("export_per_domain_csv", False))

    # fixed spacing + fixed output size (after bbox crop)
    target_spacing = cfg.get("target_spacing", [1.0, 1.0, 3.0])
    target_spacing = (float(target_spacing[0]), float(target_spacing[1]), float(target_spacing[2]))

    output_size = list(cfg.get("output_size", [144, 144, 48]))
    output_size = [int(x) for x in output_size]

    pad_value_ct = float(cfg.get("pad_value_ct", -1024.0))
    pad_value_pt = float(cfg.get("pad_value_pt", 0.0))
    pad_value_mask = float(cfg.get("pad_value_mask", 0.0))

    interp_ct = get_interpolator(cfg.get("interp_ct", "linear"))
    interp_pt = get_interpolator(cfg.get("interp_pt", "linear"))
    interp_mask = get_interpolator(cfg.get("interp_mask", "nearest"))

    save_float_dtype = cfg.get("save_float_dtype", "float32")
    save_mask_dtype = cfg.get("save_mask_dtype", "uint8")

    ct_suffix = cfg.get("ct_suffix", "_ct.nii.gz")
    pt_suffix = cfg.get("pt_suffix", "_pt.nii.gz")
    gt_suffix = cfg.get("gt_suffix", "_gtvt.nii.gz")

    # split config
    enable_split = bool(cfg.get("enable_split", False))
    seed = int(cfg.get("seed", 2026))
    val_per_center = int(cfg.get("val_per_center", 5))
    source_centers = cfg.get("source_centers", [])
    target_centers = cfg.get("target_centers", [])
    other_centers_policy = cfg.get("other_centers_policy", "ignore")

    img_out_dir = out_root / "images"
    lab_out_dir = out_root / "labels"
    ensure_dir(img_out_dir)
    ensure_dir(lab_out_dir)
    ensure_dir(out_manifest_csv.parent)

    # read csvs
    df_bbox = pd.read_csv(bbox_csv)
    required_bbox_cols = ["PatientID", "x1", "x2", "y1", "y2", "z1", "z2"]
    missing = [c for c in required_bbox_cols if c not in df_bbox.columns]
    if missing:
        raise RuntimeError(f"bbox_csv missing columns: {missing}. Found: {list(df_bbox.columns)}")

    df_info = pd.read_csv(info_csv)
    if "PatientID" not in df_info.columns:
        raise RuntimeError(f"info_csv missing 'PatientID'. Found: {list(df_info.columns)}")
    if "CenterID" not in df_info.columns:
        raise RuntimeError(f"info_csv missing 'CenterID'. Found: {list(df_info.columns)}")

    df = pd.merge(df_bbox, df_info, on="PatientID", how="inner")
    df["center_code"] = df["PatientID"].apply(patient_center_code)

    df = assign_splits(
        df,
        enable_split=enable_split,
        source_centers=source_centers,
        target_centers=target_centers,
        val_per_center=val_per_center,
        seed=seed,
        other_policy=other_centers_policy,
    )

    # ---- Mode: split_only ----
    if args.mode == "split_only":
        df_out = build_manifest_csv_only(
            df=df,
            nii_root=nii_root,
            out_root=out_root,
            out_manifest_csv=out_manifest_csv,
            export_per_domain_csv=export_per_domain_csv,
            ct_suffix=ct_suffix,
            pt_suffix=pt_suffix,
            gt_suffix=gt_suffix,
        )
        n_total = len(df)
        n_used = len(df_out)
        print(f"[SPLIT_ONLY DONE] merged_rows={n_total}, exported_rows={n_used}")
        print(f"[MANIFEST] {out_manifest_csv}")
        return

    # ---- Mode: full preprocessing ----
    rows = []
    n_total = len(df)
    n_done = 0
    n_skipped = 0

    for _, r in tqdm(df.iterrows(), total=n_total, desc="Preprocessing HECKTOR2021"):
        pid = str(r["PatientID"])
        center_code = str(r["center_code"])
        center_id = r.get("CenterID", None)
        domain = str(r.get("domain", ""))
        split = str(r.get("split", ""))

        if split == "ignore" or domain == "ignore":
            n_skipped += 1
            continue

        ct_path = nii_root / f"{pid}{ct_suffix}"
        pt_path = nii_root / f"{pid}{pt_suffix}"
        gt_path = nii_root / f"{pid}{gt_suffix}"

        if not ct_path.exists() or not pt_path.exists() or not gt_path.exists():
            rows.append({
                "patient_id": pid,
                "center_code": center_code,
                "center_id": center_id,
                "domain": domain,
                "split": split,
                "status": "missing_file",
                "ct_raw": str(ct_path),
                "pt_raw": str(pt_path),
                "gtvt_raw": str(gt_path),
            })
            n_skipped += 1
            continue

        x1, x2 = float(r["x1"]), float(r["x2"])
        y1, y2 = float(r["y1"]), float(r["y2"])
        z1, z2 = float(r["z1"]), float(r["z2"])

        try:
            # ---- read raw ----
            ct_raw = sitk.ReadImage(str(ct_path))
            pt_raw = sitk.ReadImage(str(pt_path))
            gt_raw = sitk.ReadImage(str(gt_path))

            ct_size_raw = list(ct_raw.GetSize())
            ct_spacing_raw = list(ct_raw.GetSpacing())
            pt_size_raw = list(pt_raw.GetSize())
            pt_spacing_raw = list(pt_raw.GetSpacing())

            # ---- 1) resample CT to fixed spacing (CT is the reference grid) ----
            ct = resample_to_spacing(
                img=ct_raw,
                target_spacing=target_spacing,
                interpolator=interp_ct,
                default_value=pad_value_ct,
                out_pixel_type=sitk.sitkFloat32,
            )
            ct_size_rs = list(ct.GetSize())
            ct_spacing_rs = list(ct.GetSpacing())

            # ---- 2) resample PET/GT to CT grid (no extra registration; just regrid) ----
            pt = resample_to_reference(
                moving=pt_raw,
                reference=ct,
                interpolator=interp_pt,
                default_value=pad_value_pt,
                out_pixel_type=sitk.sitkFloat32,
            )
            gt = resample_to_reference(
                moving=gt_raw,
                reference=ct,
                interpolator=interp_mask,
                default_value=pad_value_mask,
                out_pixel_type=sitk.sitkUInt8,
            )

            # ---- 3) bbox(mm) -> index ROI on CT grid ----
            start_idx, roi_size, dbg_roi = bbox_mm_to_index_roi(ct, x1, x2, y1, y2, z1, z2)

            # ---- 4) pad if bbox goes out of bounds (apply on each modality consistently) ----
            ct_pad, start_ct, dbg_pad_ct = pad_if_needed(ct, start_idx, roi_size, pad_value_ct)
            pt_pad, start_pt, dbg_pad_pt = pad_if_needed(pt, start_idx, roi_size, pad_value_pt)
            gt_pad, start_gt, dbg_pad_gt = pad_if_needed(gt, start_idx, roi_size, pad_value_mask)

            # pad_if_needed should yield the same start shift; use ct's start
            start_use = start_ct

            # ---- 5) crop ROI ----
            ct_crop = crop_roi(ct_pad, start_use, roi_size)
            pt_crop = crop_roi(pt_pad, start_use, roi_size)
            gt_crop = crop_roi(gt_pad, start_use, roi_size)

            crop_size = list(ct_crop.GetSize())  # should match for all

            # ---- 6) center pad/crop to fixed output_size (NO resize) ----
            pad_before, pad_after, crop_lower, crop_upper = compute_center_pad_crop_params(
                cur_size=crop_size,
                target_size=output_size,
            )

            ct_out_img = apply_center_pad_crop(
                ct_crop, output_size, pad_value_ct, pad_before, pad_after, crop_lower, crop_upper
            )
            pt_out_img = apply_center_pad_crop(
                pt_crop, output_size, pad_value_pt, pad_before, pad_after, crop_lower, crop_upper
            )
            gt_out_img = apply_center_pad_crop(
                gt_crop, output_size, pad_value_mask, pad_before, pad_after, crop_lower, crop_upper
            )

            # ---- 7) cast dtype ----
            ct_out_img = cast_float_dtype(ct_out_img, save_float_dtype)
            pt_out_img = cast_float_dtype(pt_out_img, save_float_dtype)
            gt_out_img = cast_mask_dtype(gt_out_img, save_mask_dtype)

            # ---- 8) write ----
            ct_out = img_out_dir / f"{pid}_ct.nii.gz"
            pt_out = img_out_dir / f"{pid}_pt.nii.gz"
            gt_out = lab_out_dir / f"{pid}_gtvt.nii.gz"

            sitk.WriteImage(ct_out_img, str(ct_out), useCompression=True)
            sitk.WriteImage(pt_out_img, str(pt_out), useCompression=True)
            sitk.WriteImage(gt_out_img, str(gt_out), useCompression=True)

            rows.append({
                "patient_id": pid,
                "center_code": center_code,
                "center_id": center_id,
                "domain": domain,
                "split": split,
                "status": "ok",

                "ct_raw": str(ct_path),
                "pt_raw": str(pt_path),
                "gtvt_raw": str(gt_path),

                "ct_proc": str(ct_out),
                "pt_proc": str(pt_out),
                "gtvt_proc": str(gt_out),

                "ct_size_raw": ",".join(map(str, ct_size_raw)),
                "ct_spacing_raw": ",".join([f"{x:.6f}" for x in ct_spacing_raw]),
                "pt_size_raw": ",".join(map(str, pt_size_raw)),
                "pt_spacing_raw": ",".join([f"{x:.6f}" for x in pt_spacing_raw]),

                "ct_size_resampled": ",".join(map(str, ct_size_rs)),
                "ct_spacing_resampled": ",".join([f"{x:.6f}" for x in ct_spacing_rs]),

                "bbox_x1": x1, "bbox_x2": x2,
                "bbox_y1": y1, "bbox_y2": y2,
                "bbox_z1": z1, "bbox_z2": z2,

                "roi_start_idx": ",".join(map(str, dbg_roi["start_idx"])),
                "roi_end_idx": ",".join(map(str, dbg_roi["end_idx"])),
                "roi_size_idx": ",".join(map(str, dbg_roi["roi_size"])),

                "pad_ct_before": ",".join(map(str, dbg_pad_ct["pad_before"])),
                "pad_ct_after": ",".join(map(str, dbg_pad_ct["pad_after"])),

                "crop_size_before_fix": ",".join(map(str, crop_size)),
                "final_output_size": ",".join(map(str, output_size)),
                "final_spacing": ",".join([f"{x:.6f}" for x in target_spacing]),
            })

            n_done += 1

        except Exception as e:
            rows.append({
                "patient_id": pid,
                "center_code": center_code,
                "center_id": center_id,
                "domain": domain,
                "split": split,
                "status": f"error:{type(e).__name__}",
                "error_msg": str(e),
                "ct_raw": str(ct_path),
                "pt_raw": str(pt_path),
                "gtvt_raw": str(gt_path),
            })
            n_skipped += 1

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_manifest_csv, index=False)

    if export_per_domain_csv and len(df_out) > 0:
        src = df_out[df_out["domain"] == "source"].copy()
        tgt = df_out[df_out["domain"] == "target"].copy()
        if len(src) > 0:
            src.to_csv(out_manifest_csv.with_name("source.csv"), index=False)
        if len(tgt) > 0:
            tgt.to_csv(out_manifest_csv.with_name("target.csv"), index=False)

    print(f"[DONE] processed={n_done}, skipped={n_skipped}, total_in_merged_csv={n_total}")
    print(f"[MANIFEST] {out_manifest_csv}")


if __name__ == "__main__":
    main()