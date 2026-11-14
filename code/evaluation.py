"""evaluation.py: scoring utilities for the FLAIR track-validation project (DS340W).

This module is support infrastructure; it is NOT the novel method (see track_validation.py).

What it does
------------
* Decodes the authors' ground-truth (GT) PNG masks (two colour-coded sharks).
* Loads predicted masks from either schema we have to deal with and normalises both to
  ``{track_id: boolean (H, W) array}`` per frame:
    - released masks:   .npy pickled dict, values are boolean arrays shaped (1, H, W) or (H, W)
    - our notebooks:    .npy pickled dict, values are lists of polygons (decoded with polygons_to_mask)
* Computes, per video (never pooled):
    - authors_dice   best-match Dice (each GT shark scored by its best-matching predicted mask,
                     averaged over sharks, then frames). Missing prediction file counts as 0.
                     The released script has a zero-division bug; here empty/empty returns 0.0.
    - union_dice     Dice of (all kept masks combined) vs (all GT sharks combined). Our own metric.
                     It penalises extra area, including overlapping masks on a real shark.
    - tracks_present / tracks_retained / shark_tracks_retained / extra_tracks_retained
    - recall         GT shark instance "found" if any retained mask has Dice >= FOUND_DICE with it.

Definitions (frozen in PROTOCOL.md before any new-method results)
-----------------------------------------------------------------
* A track is GT-associated on an annotated frame if >= SHARK_AREA_FRACTION (0.5) of its mask area
  lies inside the GT sharks of that frame.
* A track is shark-associated if it is GT-associated on >= SHARK_FRAME_FRACTION (0.5) of the annotated
  frames where it has a non-empty mask. Extra track = present but not shark-associated.
* ``oracle_keep`` keeps exactly the shark-associated tracks (a GT-dependent reference, not deployable).

Ground truth is sparse annotated frames, not identity-level trajectories, so everything here is
"overlaps a GT shark on annotated frames", nothing stronger.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

# --------------------------------------------------------------------------- constants
GT_COLORS = ((250, 50, 83), (184, 61, 245))

# Clip start frames (GT frame numbers use full-video numbering; most predicted masks are clip-relative).
# Mask R-CNN folders use full-video numbering: pass full_video_numbering=True for those.
VIDEO_START = {
    "video1": 2400, "video2": 5160, "video3": 2550, "video4": 1050,
    "video5": 4920, "video6": 2640, "video7": 7200,
    "blacktip": 0, "white1": 0,
}

SHARK_AREA_FRACTION = 0.5
SHARK_FRAME_FRACTION = 0.5
FOUND_DICE = 0.5


# --------------------------------------------------------------------------- basic metrics
def dice(a, b) -> float:
    """Dice coefficient of two boolean masks. Returns 0.0 if both are empty (zero-division guard)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def decode_gt(rgb: np.ndarray) -> List[np.ndarray]:
    """Boolean mask per GT colour that is present in a colour-coded GT image (H, W, 3)."""
    rgb = np.asarray(rgb)[..., :3]
    out = []
    for color in GT_COLORS:
        m = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
        if m.any():
            out.append(m)
    return out


# --------------------------------------------------------------------------- loading / schema
def polygons_to_mask(polygons, image_shape) -> np.ndarray:
    """Same as the notebooks' polygons_to_mask (cv2.fillPoly on integer polygons)."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("opencv-python is needed to decode polygon masks") from e
    mask = np.zeros(tuple(image_shape[:2]), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def to_mask_dict(raw: dict, image_shape=None, pack: bool = False) -> Dict[int, np.ndarray]:
    """Normalise one frame's raw dict to {track_id: boolean (H, W) mask}.

    Values may be boolean arrays ((1, H, W) or (H, W)) or lists of polygons (needs image_shape).
    pack=True stores each mask as a PackedMask (8x less memory, array-like).
    """
    out: Dict[int, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            if image_shape is None:
                raise ValueError("Polygon masks need image_shape=(H, W) to be decoded")
            m = polygons_to_mask(value, image_shape)
        else:
            m = np.squeeze(np.asarray(value)).astype(bool)
        out[key] = PackedMask(m) if pack else m
    return out


def load_pred_frame(path: str, image_shape=None, pack: bool = False) -> Optional[Dict[int, np.ndarray]]:
    """Load one frame_*.npy as {track_id: bool mask}; None if the file does not exist."""
    if not os.path.exists(path):
        return None
    raw = np.load(path, allow_pickle=True).item()
    return to_mask_dict(raw, image_shape, pack)


def parse_gt_name(name: str, default_video: Optional[str] = None):
    """'video1_02589.png' -> ('video1', 2589); '0047.png' -> (default_video, 47).

    blacktip/white1 GT masks have no video prefix, so pass default_video for those folders.
    Raises ValueError on an unexpected name.
    """
    base = os.path.basename(name)
    m = re.match(r"^(?P<video>.+?)_(?P<frame>\d+)\.(png|jpg|jpeg)$", base, re.I)
    if m:
        return m.group("video"), int(m.group("frame"))
    m = re.match(r"^(?P<frame>\d+)\.(png|jpg|jpeg)$", base, re.I)
    if m and default_video:
        return default_video, int(m.group("frame"))
    raise ValueError(f"Unrecognised GT filename: {name}")


def pred_candidates(video: str, frame: int) -> List[str]:
    """Filenames a predicted-mask frame may have (checked in order).

    Seen in the released data: video1_frame_1143.npy (FLAIR _3), frame_1143.npy (most others),
    02589.npy (Mask R-CNN, zero-padded full-video numbering).
    """
    return [f"{video}_frame_{frame}.npy", f"frame_{frame}.npy", f"{frame:05d}.npy", f"{frame:04d}.npy"]


def image_candidates(video: str, frame: int) -> List[str]:
    """Filenames the annotated-frame image may have: video1_02589.jpg, frame_0047.jpg, 0047.jpg ..."""
    return [f"{video}_{frame:05d}.jpg", f"frame_{frame:04d}.jpg", f"{frame:04d}.jpg", f"{frame:05d}.jpg",
            f"{video}_{frame}.jpg", f"frame_{frame}.jpg"]


def find_image(image_dir: str, video: str, frame: int) -> Optional[str]:
    for name in image_candidates(video, frame):
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return path
    # Fallback: any jpg/jpeg/png whose last number in the name equals the frame (e.g. frame_73.png, white1_0073.JPG).
    try:
        names = sorted(os.listdir(image_dir))
    except OSError:
        return None
    for name in names:
        m = re.match(r"^.*?(\d+)\.(jpg|jpeg|png)$", name, re.I)
        if m and int(m.group(1)) == frame:
            return os.path.join(image_dir, name)
    return None


class PackedMask:
    """Boolean mask stored bit-packed (8x smaller). Behaves like a read-only boolean array.

    A 3840x2160 mask is ~8 MB as bool but ~1 MB packed; this keeps a 27-track video in memory.
    """
    __slots__ = ("_bits", "shape", "_area")

    def __init__(self, mask):
        a = np.squeeze(np.asarray(mask)).astype(bool)
        self.shape = a.shape
        self._bits = np.packbits(a.ravel())
        self._area = int(a.sum())

    def __array__(self, dtype=None, copy=None):
        n = self.shape[0] * self.shape[1]
        a = np.unpackbits(self._bits, count=n).reshape(self.shape).astype(bool)
        return a if dtype is None else a.astype(dtype)

    def sum(self, *args, **kwargs):
        if not args and not kwargs:
            return self._area
        return np.asarray(self).sum(*args, **kwargs)

    def any(self):
        return self._area > 0


def find_pred_file(pred_dir: str, video: str, frame: int) -> str:
    for name in pred_candidates(video, frame):
        path = os.path.join(pred_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(pred_dir, pred_candidates(video, frame)[0])  # non-existent -> counts as missing


@dataclass
class FrameRecord:
    """One annotated frame: GT sharks and (normalised) predicted masks. pred=None means missing file."""
    video: str
    gt_frame: int
    gt: List[np.ndarray]
    pred: Optional[Dict[int, np.ndarray]]


def build_records(video: str, gt_dir: str, pred_dir: str, image_shape=None,
                  start: Optional[int] = None, full_video_numbering: bool = False,
                  limit: Optional[int] = None, pack: bool = True) -> List[FrameRecord]:
    """Pair every GT PNG of `video` in gt_dir with its predicted-mask file in pred_dir.

    clip frame = GT frame - start (start defaults to VIDEO_START[video]); pass full_video_numbering=True
    when the predictions use full-video frame numbers (e.g. Mask R-CNN). Predicted masks are bit-packed
    by default so a many-track 4K video fits in memory.
    """
    from PIL import Image

    if start is None:
        start = VIDEO_START[video]
    names = []
    for n in sorted(os.listdir(gt_dir)):
        try:
            v, f = parse_gt_name(n, default_video=video)
        except ValueError:
            continue
        if v == video:
            names.append((f, n))
    names.sort()
    if limit:
        names = names[:limit]

    records = []
    for gt_frame, n in names:
        gt = decode_gt(np.array(Image.open(os.path.join(gt_dir, n)).convert("RGB")))
        clip_frame = gt_frame if full_video_numbering else gt_frame - start
        pred = load_pred_frame(find_pred_file(pred_dir, video, clip_frame), image_shape, pack=pack)
        records.append(FrameRecord(video, gt_frame, gt, pred))
    return records


# --------------------------------------------------------------------------- track labels
def label_tracks(records: Sequence[FrameRecord]) -> Dict[int, dict]:
    """Shark-associated labels per track (see module docstring). Uses all predicted tracks."""
    stats: Dict[int, dict] = {}
    for rec in records:
        if rec.pred is None or not rec.gt:
            continue
        gt_union = np.any(np.stack(rec.gt), axis=0)
        for tid, m in rec.pred.items():
            area = int(m.sum())
            if area == 0:
                continue
            s = stats.setdefault(tid, {"n_frames": 0, "n_associated": 0})
            s["n_frames"] += 1
            if np.logical_and(m, gt_union).sum() / area >= SHARK_AREA_FRACTION:
                s["n_associated"] += 1
    for s in stats.values():
        s["shark"] = (s["n_associated"] / s["n_frames"]) >= SHARK_FRAME_FRACTION
    return stats


def oracle_keep(labels: Dict[int, dict]) -> Set[int]:
    """GT-dependent reference only (not deployable): keep shark-associated tracks."""
    return {tid for tid, s in labels.items() if s["shark"]}


# --------------------------------------------------------------------------- scoring
def evaluate_records(records: Sequence[FrameRecord], keep: Optional[Iterable[int]] = None,
                     labels: Optional[Dict[int, dict]] = None) -> dict:
    """Score one video. `keep` = track ids retained (None = keep every predicted track).

    Labels are always computed from the full (unpruned) prediction so extra/shark counts are comparable
    across methods.
    """
    keep_set = None if keep is None else set(keep)
    if labels is None:
        labels = label_tracks(records)

    frame_authors, frame_union = [], []
    n_gt = n_found = n_missing = n_no_gt = 0

    for rec in records:
        if not rec.gt:
            n_no_gt += 1
            continue
        n_gt += len(rec.gt)
        if rec.pred is None:
            n_missing += 1
            frame_authors.append(0.0)
            frame_union.append(0.0)
            continue
        kept = {t: m for t, m in rec.pred.items() if keep_set is None or t in keep_set}
        best = [max((dice(g, m) for m in kept.values()), default=0.0) for g in rec.gt]
        frame_authors.append(float(np.mean(best)))
        n_found += sum(b >= FOUND_DICE for b in best)
        gt_union = np.any(np.stack(rec.gt), axis=0)
        pred_union = np.any(np.stack(list(kept.values())), axis=0) if kept else np.zeros_like(gt_union)
        frame_union.append(dice(gt_union, pred_union))

    present = set(labels)
    retained = present if keep_set is None else present & keep_set
    shark_retained = {t for t in retained if labels[t]["shark"]}
    return {
        "frames_scored": len(frame_authors),
        "missing_frames": n_missing,
        "frames_without_gt": n_no_gt,
        "authors_dice": float(np.mean(frame_authors)) if frame_authors else float("nan"),
        "union_dice": float(np.mean(frame_union)) if frame_union else float("nan"),
        "gt_instances": n_gt,
        "gt_found": int(n_found),
        "recall": (n_found / n_gt) if n_gt else float("nan"),
        "tracks_present": len(present),
        "tracks_retained": len(retained),
        "shark_tracks_retained": len(shark_retained),
        "extra_tracks_retained": len(retained) - len(shark_retained),
    }


def gate_check(baseline: dict, method: dict, extra_removed_min: float = 0.5, dice_drop_max: float = 0.01) -> dict:
    """Pre-specified gate (PROTOCOL.md): removes >= 50% of extra tracks, recall unchanged vs baseline,
    best-match Dice drops by at most 0.01."""
    extra0, extra1 = baseline["extra_tracks_retained"], method["extra_tracks_retained"]
    removed = (extra0 - extra1) / extra0 if extra0 else float("nan")
    ok_extra = (extra0 > 0) and (removed >= extra_removed_min)
    ok_recall = method["recall"] >= baseline["recall"] - 1e-12
    ok_dice = (baseline["authors_dice"] - method["authors_dice"]) <= dice_drop_max + 1e-12
    return {"extra_removed_frac": removed, "ok_extra": bool(ok_extra), "ok_recall": bool(ok_recall),
            "ok_dice": bool(ok_dice), "passed": bool(ok_extra and ok_recall and ok_dice)}


# --------------------------------------------------------------------------- self-test against the released scores
def check_against_summary(computed: Dict[tuple, dict], summary_csv: str, tol: float = 1e-3) -> List[str]:
    """Compare computed {(method, video): metrics} with dice_summary.csv (dice_authors, dice_union).

    Returns a list of mismatch messages (empty list = the scorer reproduces the released numbers).
    """
    problems = []
    with open(summary_csv, newline="") as f:
        rows = {(r["method"], r["video"]): r for r in csv.DictReader(f)}
    for key, metrics in computed.items():
        if key not in rows:
            problems.append(f"{key}: not in summary csv")
            continue
        for ours, theirs in (("authors_dice", "dice_authors"), ("union_dice", "dice_union")):
            ref = float(rows[key][theirs])
            if abs(metrics[ours] - ref) > tol:
                problems.append(f"{key} {ours}: ours {metrics[ours]:.4f} vs csv {ref:.4f}")
    return problems
