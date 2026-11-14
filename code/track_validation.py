"""track_validation.py: automatic track-level CLIP validation (the novel block).

Replaces FLAIR's optional manual pruning (KEEP_KEYS) with an automatic rule:

    confirmed tracks -> sample frames -> CLIP shark probability per crop -> aggregate over the track
    -> keep the track if the aggregate >= threshold

The SAME function is used by the offline released-mask experiments and by notebook 04, so the experiment
and the pipeline cannot drift apart. Nothing here depends on torch at import time; the CLIP model is
only built by ``make_clip_scorer`` and any callable can be injected for testing.

Frozen choices (PROTOCOL.md): up to 10 evenly spaced non-empty frames per track (all if fewer);
FLAIR's crop (mask box + 25 px padding, clipped to the image); FLAIR's CLIP model, prompts and
preprocessing; aggregates = median (primary) and 25th percentile; keep if aggregate >= 0.95. The fraction of
crops >= 0.95 is recorded as a diagnostic only. 0.95 was FLAIR's per-candidate threshold, not a justified
track-level cutoff: it is fixed, not tuned.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# FLAIR's CLIP setup (copied from notebooks 02/03 so the validator scores crops the same way)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_PROMPTS = [
    "A drone image of white shape on surface of the ocean",
    "A clear zoomed-in drone image of a complete dark nurse shark in the ocean with distinct pectoral fins and tail",
    "A drone image of the ocean with no distinct sharks",
    "A drone image of a rock under the surface of the ocean with no sharks",
    "A drone image of sand in the ocean",
    "A drone image of rocks on land",
]
CLIP_CORRECT = [1]
BBOX_BUFFER = 25
THRESHOLD = 0.95
MAX_OBS = 10

VARIANTS = {"median_clip": "median", "q25_clip": "q25"}


# --------------------------------------------------------------------------- crops and sampling
def mask_box(mask: np.ndarray):
    """(x_min, y_min, x_max, y_max) of a boolean mask (inclusive), or None if empty."""
    ys, xs = np.where(np.squeeze(mask))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_for_mask(image: np.ndarray, mask: np.ndarray, buffer: int = BBOX_BUFFER) -> Optional[np.ndarray]:
    """FLAIR-style crop: mask box padded by `buffer` px and clipped to the image; None if the mask is empty.

    Mirrors the notebooks (box clipped to the image, crop = img[y1:y2+1, x1:x2+1]).
    """
    box = mask_box(mask)
    if box is None:
        return None
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    X1, Y1 = max(0, x0 - buffer), max(0, y0 - buffer)
    X2, Y2 = min(x1 + buffer, w), min(y1 + buffer, h)
    return image[Y1:Y2 + 1, X1:X2 + 1]


def sample_frames(frames: Sequence[int], max_obs: int = MAX_OBS) -> List[int]:
    """Up to `max_obs` evenly spaced frames (all of them if fewer)."""
    frames = sorted(frames)
    if len(frames) <= max_obs:
        return frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, max_obs)).astype(int))
    return [frames[i] for i in idx]


# --------------------------------------------------------------------------- aggregation
def aggregate(probs: Sequence[float], how: str) -> float:
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return float("nan")
    if how == "median":
        return float(np.median(probs))
    if how == "q25":
        return float(np.percentile(probs, 25))
    raise ValueError(f"unknown aggregate: {how}")


def score_tracks(track_frames: Dict[int, Sequence[int]],
                 get_mask: Callable[[int, int], np.ndarray],
                 get_image: Callable[[int], np.ndarray],
                 scorer: Callable[[List[np.ndarray]], Sequence[float]],
                 max_obs: int = MAX_OBS,
                 buffer: int = BBOX_BUFFER,
                 threshold: float = THRESHOLD) -> List[dict]:
    """Score every track. One row per track:
    track_id, n_obs, probs, median, q25, frac_ge_095 (diagnostic), keep_median_clip, keep_q25_clip.

    track_frames : {track_id: frames where the track has a non-empty mask}
    get_mask     : (track_id, frame) -> boolean mask
    get_image    : frame -> RGB image (H, W, 3)
    scorer       : list of crops -> shark probabilities (one per crop)
    """
    rows = []
    for tid, frames in sorted(track_frames.items()):
        chosen = sample_frames(frames, max_obs)
        crops, used = [], []
        for f in chosen:
            crop = crop_for_mask(get_image(f), get_mask(tid, f), buffer)
            if crop is not None and crop.size:
                crops.append(crop)
                used.append(f)
        probs = [float(p) for p in scorer(crops)] if crops else []
        med, q25 = aggregate(probs, "median"), aggregate(probs, "q25")
        rows.append({
            "track_id": tid,
            "n_obs": len(probs),
            "frames": used,
            "probs": probs,
            "median": med,
            "q25": q25,
            "frac_ge_095": float(np.mean(np.asarray(probs) >= 0.95)) if probs else float("nan"),
            "keep_median_clip": bool(probs) and med >= threshold,
            "keep_q25_clip": bool(probs) and q25 >= threshold,
        })
    return rows


def validate_tracks(track_frames, get_mask, get_image, scorer, variant: str = "median_clip",
                    max_obs: int = MAX_OBS, buffer: int = BBOX_BUFFER, threshold: float = THRESHOLD):
    """Run the automatic validator. Returns (kept_track_ids, per_track_rows).

    Tracks with no usable crop are dropped (nothing to score). Tracks with fewer than max_obs observations
    use all of them; a one-observation track just reproduces FLAIR's single-frame score (a caveat).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    rows = score_tracks(track_frames, get_mask, get_image, scorer, max_obs, buffer, threshold)
    key = "keep_" + variant
    kept = sorted(r["track_id"] for r in rows if r[key])
    return kept, rows


def select_tracks(track_filter: str, manual_keep: Sequence[int], **validate_kwargs):
    """Single switch used by notebook 04:  TRACK_FILTER = "manual" | "median_clip" | "q25_clip"."""
    if track_filter == "manual":
        return sorted(manual_keep), None
    return validate_tracks(variant=track_filter, **validate_kwargs)


# --------------------------------------------------------------------------- the real CLIP scorer
def pad_to_square(crop: np.ndarray) -> np.ndarray:
    """Pad a crop to a square with its mean colour, centred (no resizing, no content removed).

    EXPLORATORY ONLY: FLAIR feeds crops straight to CLIP's default preprocessing (resize shorter side,
    then centre-crop), which trims the ends of tall/narrow crops. This is used only to test that idea.
    """
    h, w = crop.shape[:2]
    if h == w:
        return crop
    side = max(h, w)
    fill = crop.reshape(-1, crop.shape[-1]).mean(axis=0).astype(crop.dtype)
    canvas = np.empty((side, side, crop.shape[-1]), dtype=crop.dtype)
    canvas[:] = fill
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = crop
    return canvas


def load_clip(device: Optional[str] = None) -> dict:
    """Load FLAIR's CLIP model/processor once; pass the result as `bundle=` to make_clip_scorer."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=False)
    return {"model": model, "processor": processor, "device": device}


def make_clip_scorer(device: Optional[str] = None, prompts: Sequence[str] = CLIP_PROMPTS,
                     correct: Sequence[int] = CLIP_CORRECT, batch_size: int = 16,
                     pad_square: bool = False, bundle: Optional[dict] = None):
    """Build FLAIR's CLIP scorer: crops -> shark probability (max over the 'correct' prompt indices).

    Same model, prompts and softmax-over-prompts as notebooks 02/03. pad_square=True is the exploratory
    variant (see pad_to_square); the default reproduces FLAIR's preprocessing exactly.
    """
    import torch
    from PIL import Image

    b = bundle or load_clip(device)
    model, processor, device = b["model"], b["processor"], b["device"]

    def scorer(crops: List[np.ndarray]) -> List[float]:
        out: List[float] = []
        for i in range(0, len(crops), batch_size):
            batch = []
            for c in crops[i:i + batch_size]:
                if not isinstance(c, Image.Image):
                    c = np.ascontiguousarray(pad_to_square(c) if pad_square else c)
                    c = Image.fromarray(c)
                batch.append(c)
            inputs = processor(text=list(prompts), images=batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                probs = model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
            out.extend(float(p[list(correct)].max()) for p in probs)
        return out

    return scorer
