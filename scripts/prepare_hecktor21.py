# prep_hecktor21.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Any

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
    """
    HECKTOR patient ids typically start with 4-letter center code, e.g. CHGJ007 -> CHGJ.
    """
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


def bbox_mm_to_index_roi(
    img: sitk.Image,
    x1: float, x2: float,
    y1: float, y2: float,
    z1: float, z2: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Convert a physical-space bbox (mm) into an axis-aligned index ROI on img grid.
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
        # continuous index in image grid
        ci = img.TransformPhysicalPointToContinuousIndex(p)
        idxs.append(ci)

    idxs = np.array(idxs, dtype=np.float64)  # (8, 3)
    mins = idxs.min(axis=0)
    maxs = idxs.max(axis=0)

    start = np.floor(mins).astype(int)
    end = np.ceil(maxs).astype(int)  # end index (inclusive-ish); we'll treat as inclusive bound

    # We want an inclusive range [start, end], thus size = end-start+1
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
    """
    If ROI extends outside image bounds, constant-pad image so ROI is valid.
    Returns padded_img, new_start_idx, pad_debug
    """
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


def resize_to_target(
    img: sitk.Image,
    target_size: List[int],
    interpolator: int,
    default_value: float,
    out_pixel_type: int,
) -> Tuple[sitk.Image, List[float]]:
    """
    Resample img to target_size while preserving physical FOV of current img.
    """
    in_size = np.array(list(img.GetSize()), dtype=np.float64)
    in_spacing = np.array(list(img.GetSpacing()), dtype=np.float64)
    fov_mm = in_size * in_spacing  # physical extent covered by the image grid (approx)

    target_size_np = np.array(target_size, dtype=np.float64)
    new_spacing = (fov_mm / target_size_np).tolist()

    ref = sitk.Image([int(x) for x in target_size], out_pixel_type)
    ref.SetSpacing([float(s) for s in new_spacing])
    ref.SetDirection(img.GetDirection())
    ref.SetOrigin(img.GetOrigin())

    out = sitk.Resample(img, ref, sitk.Transform(), interpolator, default_value, out_pixel_type)
    return out, new_spacing


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
    Adds columns domain, split.
    - target -> test
    - source -> train/val with val_per_center per center
    - others -> per policy
    """
    df = df.copy()
    if not enable_split:
        df["domain"] = "all"
        df["split"] = "train"
        return df

    source_centers = [c.upper() for c in source_centers]
    target_centers = [c.upper() for c in target_centers]
    other_policy = other_policy.lower().strip()

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

    # default split
    df["split"] = "ignore"
    df.loc[df["domain"] == "target", "split"] = "test"
    df.loc[df["domain"] == "source", "split"] = "train"

    # center-stratified internal validation within source
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
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--workers", type=int, default=1, help="Reserved for future; currently single-process.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    bbox_csv = cfg["bbox_csv"]
    info_csv = cfg["info_csv"]
    nii_root = Path(cfg["nii_root"])

    out_root = Path(cfg["out_root"])
    out_manifest_csv = Path(cfg["out_manifest_csv"])
    export_per_domain_csv = bool(cfg.get("export_per_domain_csv", False))

    target_size = list(cfg.get("target_size", [144, 144, 144]))
    pad_value_ct = float(cfg.get("pad_value_ct", -1024.0))
    pad_value_pt = float(cfg.get("pad_value_pt", 0.0))
    pad_value_mask = float(cfg.get("pad_value_mask", 0))

    interp_img = get_interpolator(cfg.get("interp_img", "linear"))
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

    # output dirs
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

    # merge by PatientID
    df = pd.merge(df_bbox, df_info, on="PatientID", how="inner")
    df["center_code"] = df["PatientID"].apply(patient_center_code)

    # assign split/domain
    df = assign_splits(
        df,
        enable_split=enable_split,
        source_centers=source_centers,
        target_centers=target_centers,
        val_per_center=val_per_center,
        seed=seed,
        other_policy=other_centers_policy,
    )

    # process cases
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

        # skip ignored
        if split == "ignore" or domain == "ignore":
            n_skipped += 1
            continue

        ct_path = nii_root / f"{pid}{ct_suffix}"
        pt_path = nii_root / f"{pid}{pt_suffix}"
        gt_path = nii_root / f"{pid}{gt_suffix}"

        if not ct_path.exists() or not pt_path.exists() or not gt_path.exists():
            # keep a record, but skip processing
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

        # bbox in mm
        x1, x2 = float(r["x1"]), float(r["x2"])
        y1, y2 = float(r["y1"]), float(r["y2"])
        z1, z2 = float(r["z1"]), float(r["z2"])

        try:
            # read
            ct = sitk.ReadImage(str(ct_path))
            pt = sitk.ReadImage(str(pt_path))
            gt = sitk.ReadImage(str(gt_path))

            ct_size_raw = list(ct.GetSize())
            ct_spacing_raw = list(ct.GetSpacing())
            pt_size_raw = list(pt.GetSize())
            pt_spacing_raw = list(pt.GetSpacing())

            # resample PET + mask to CT grid
            pt_on_ct = resample_to_reference(
                moving=pt,
                reference=ct,
                interpolator=interp_img,
                default_value=pad_value_pt,
                out_pixel_type=sitk.sitkFloat32,
            )
            gt_on_ct = resample_to_reference(
                moving=gt,
                reference=ct,
                interpolator=interp_mask,
                default_value=pad_value_mask,
                out_pixel_type=sitk.sitkUInt8,
            )

            # bbox(mm) -> index ROI on CT grid
            start_idx, roi_size, dbg_roi = bbox_mm_to_index_roi(ct, x1, x2, y1, y2, z1, z2)

            # pad if needed for each modality
            ct_pad, start_ct, dbg_pad_ct = pad_if_needed(ct, start_idx, roi_size, pad_value_ct)
            pt_pad, start_pt, dbg_pad_pt = pad_if_needed(pt_on_ct, start_idx, roi_size, pad_value_pt)
            gt_pad, start_gt, dbg_pad_gt = pad_if_needed(gt_on_ct, start_idx, roi_size, pad_value_mask)

            # after padding, start index should match (we pad based on same start/roi)
            # but to be safe, we use start_ct for all (they should be identical)
            start_use = start_ct

            # crop
            ct_crop = crop_roi(ct_pad, start_use, roi_size)
            pt_crop = crop_roi(pt_pad, start_use, roi_size)
            gt_crop = crop_roi(gt_pad, start_use, roi_size)

            # resize to target_size (preserve cropped FOV)
            ct_rs, out_spacing = resize_to_target(
                ct_crop, target_size, interp_img, pad_value_ct, sitk.sitkFloat32
            )
            pt_rs, _ = resize_to_target(
                pt_crop, target_size, interp_img, pad_value_pt, sitk.sitkFloat32
            )
            gt_rs, _ = resize_to_target(
                gt_crop, target_size, interp_mask, pad_value_mask, sitk.sitkUInt8
            )

            # cast dtype
            ct_rs = cast_float_dtype(ct_rs, save_float_dtype)
            pt_rs = cast_float_dtype(pt_rs, save_float_dtype)
            gt_rs = cast_mask_dtype(gt_rs, save_mask_dtype)

            # write output
            ct_out = img_out_dir / f"{pid}_ct.nii.gz"
            pt_out = img_out_dir / f"{pid}_pt.nii.gz"
            gt_out = lab_out_dir / f"{pid}_gtvt.nii.gz"

            sitk.WriteImage(ct_rs, str(ct_out), useCompression=True)
            sitk.WriteImage(pt_rs, str(pt_out), useCompression=True)
            sitk.WriteImage(gt_rs, str(gt_out), useCompression=True)

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

                "bbox_x1": x1, "bbox_x2": x2,
                "bbox_y1": y1, "bbox_y2": y2,
                "bbox_z1": z1, "bbox_z2": z2,

                "roi_start_idx": ",".join(map(str, dbg_roi["start_idx"])),
                "roi_end_idx": ",".join(map(str, dbg_roi["end_idx"])),
                "roi_size_idx": ",".join(map(str, dbg_roi["roi_size"])),

                "pad_ct_before": ",".join(map(str, dbg_pad_ct["pad_before"])),
                "pad_ct_after": ",".join(map(str, dbg_pad_ct["pad_after"])),

                "target_size": ",".join(map(str, target_size)),
                "proc_spacing": ",".join([f"{x:.6f}" for x in out_spacing]),
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

    # save manifest
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_manifest_csv, index=False)

    # optional exports
    if export_per_domain_csv and len(df_out) > 0:
        src = df_out[df_out["domain"] == "source"].copy()
        tgt = df_out[df_out["domain"] == "target"].copy()
        if len(src) > 0:
            src.to_csv(out_manifest_csv.with_name("source.csv"), index=False)
        if len(tgt) > 0:
            tgt.to_csv(out_manifest_csv.with_name("target.csv"), index=False)

    # simple summary
    print(f"[DONE] processed={n_done}, skipped={n_skipped}, total_in_merged_csv={n_total}")
    print(f"[MANIFEST] {out_manifest_csv}")


if __name__ == "__main__":
    main()