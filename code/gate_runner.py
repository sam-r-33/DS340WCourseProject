"""gate_runner.py: runs the scorer self-test and the offline CLIP gate on the released masks.

Uses the folder layout of the unzipped Zenodo data (UnzippedData/):
  trimmed_videos/sampled_frames/masks/<...>_masks/        GT PNG masks (video1_02589.png, 0047.png for blacktip)
  trimmed_videos/sampled_frames/images/<...>_sample(s)/    GT frame images (video1_02589.jpg, frame_0047.jpg)
  predicted_masks/FLAIR/video_segments_video5_small_3/     released FLAIR masks (clip-relative frame numbers)
  predicted_masks/HiL/video_segments_HiL_video5_small/     released human-in-the-loop masks

Typical Colab use is in README_gate.txt. Nothing here modifies any data; results are written to out_dir.
"""
from __future__ import annotations

import csv
import functools
import json
import os
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

import evaluation as ev
import track_validation as tv

SAMPLED = os.path.join("trimmed_videos", "sampled_frames")
_V12 = ("video1_video2_sample_masks", "seg_gt_video1_video2_sample")
_V37 = ("video3_video4_video5_video6_video7_sample_masks", "seg_gt_video3_video4_video5_video6_video7_sample")
LAYOUT = {
    "video1": _V12, "video2": _V12,
    "video3": _V37, "video4": _V37, "video5": _V37, "video6": _V37, "video7": _V37,
    "blacktip": ("blacktip_masks", "blacktip_samples"),
    "white1": ("white1_masks", "white1_samples"),
}
SINGLE_RUN = ("blacktip", "white1")   # one FLAIR run, folder has no "_small_3" suffix


def gt_dir(root: str, video: str) -> str:
    return os.path.join(root, SAMPLED, "masks", LAYOUT[video][0])


def img_dir(root: str, video: str) -> str:
    return os.path.join(root, SAMPLED, "images", LAYOUT[video][1])


def flair_dir(root: str, video: str) -> str:
    base = os.path.join(root, "predicted_masks", "FLAIR")
    names = sorted(os.listdir(base))
    if video in SINGLE_RUN:
        want = [d for d in names if d == f"video_segments_{video}"] or [d for d in names if video in d]
    else:
        want = [d for d in names if d == f"video_segments_{video}_small_3"]
    if not want:
        raise FileNotFoundError(f"No FLAIR prediction folder for {video} under {base}. Found: {names}")
    return os.path.join(base, want[0])


def hil_dir(root: str, video: str) -> str:
    base = os.path.join(root, "predicted_masks", "HiL")
    names = sorted(os.listdir(base))
    want = f"video_segments_HiL_{video}" if video in SINGLE_RUN else f"video_segments_HiL_{video}_small"
    hit = [d for d in names if d == want] or [d for d in names if video in d]
    if not hit:
        raise FileNotFoundError(f"No HiL folder for {video} under {base}. Found: {names}")
    return os.path.join(base, hit[0])


# --------------------------------------------------------------------------- 1) scorer self-test
def self_test(root: str, summary_csv: str, videos, with_hil: bool = True, tol: float = 1e-3):
    """Recompute the released HiL / FLAIR scores with evaluation.py and compare with dice_summary.csv.

    Returns (computed, problems). problems == [] means the scorer reproduces the released numbers.
    """
    computed = {}
    for v in videos:
        flair_key = "FLAIR_single" if v in SINGLE_RUN else "FLAIR_3"
        recs = ev.build_records(v, gt_dir(root, v), flair_dir(root, v))
        computed[(flair_key, v)] = ev.evaluate_records(recs)
        del recs
        if with_hil:
            recs = ev.build_records(v, gt_dir(root, v), hil_dir(root, v))
            computed[("HiL", v)] = ev.evaluate_records(recs)
            del recs
        for k in [k for k in computed if k[1] == v]:
            m = computed[k]
            print(f"{k[0]:13s} {v:9s} authors={m['authors_dice']:.4f} union={m['union_dice']:.4f} "
                  f"frames={m['frames_scored']} tracks={m['tracks_present']}")
    problems = ev.check_against_summary(computed, summary_csv, tol)
    print("\nSELF-TEST:", "PASS (matches dice_summary.csv)" if not problems else "MISMATCHES")
    for p in problems:
        print("  ", p)
    return computed, problems


# --------------------------------------------------------------------------- 2) the CLIP gate
def run_video(video: str, root: str, scorer: Callable, out_dir: Optional[str] = None,
              max_obs: int = tv.MAX_OBS, cache_frames: int = 48) -> dict:
    """Offline gate for one video on the released FLAIR masks: score tracks with the shared validator, apply
    median_clip and q25_clip, and report baseline / oracle / variants with the pre-specified gate."""
    recs = ev.build_records(video, gt_dir(root, video), flair_dir(root, video))
    labels = ev.label_tracks(recs)
    base = ev.evaluate_records(recs, labels=labels)
    oracle = ev.evaluate_records(recs, keep=ev.oracle_keep(labels), labels=labels)

    track_frames: Dict[int, list] = {}
    for idx, rec in enumerate(recs):
        if rec.pred is None:
            continue
        for tid, m in rec.pred.items():
            if m.sum() > 0:
                track_frames.setdefault(tid, []).append(idx)

    imgs = img_dir(root, video)

    # Some folders (white1) have GT masks for frames with no image in the samples folder: score tracks only on
    # frames whose image exists, and say so.
    n_missing = 0
    for tid in list(track_frames):
        have = [i for i in track_frames[tid] if ev.find_image(imgs, video, recs[i].gt_frame) is not None]
        n_missing += len(track_frames[tid]) - len(have)
        if have:
            track_frames[tid] = have
        else:
            del track_frames[tid]
    if n_missing:
        print(f"[{video}] note: {n_missing} track-frame(s) skipped because no image exists for that frame")

    @functools.lru_cache(maxsize=cache_frames)
    def get_image(idx):
        path = ev.find_image(imgs, video, recs[idx].gt_frame)
        if path is None:
            raise FileNotFoundError(f"No image for {video} frame {recs[idx].gt_frame} in {imgs}")
        return np.array(Image.open(path).convert("RGB"))

    # one-time sanity check: image and masks must have the same size
    if track_frames:
        t0 = next(iter(track_frames))
        i0 = track_frames[t0][0]
        mshape, ishape = tuple(recs[i0].pred[t0].shape), get_image(i0).shape[:2]
        if mshape != ishape:
            raise ValueError(f"{video}: mask shape {mshape} != image shape {ishape}")

    rows = tv.score_tracks(track_frames, lambda t, i: recs[i].pred[t], get_image, scorer, max_obs)
    for r in rows:
        r["shark"] = bool(labels.get(r["track_id"], {}).get("shark", False))

    result = {"video": video, "baseline": base, "oracle": oracle, "tracks": rows}
    for variant in tv.VARIANTS:
        kept = [r["track_id"] for r in rows if r["keep_" + variant]]
        m = ev.evaluate_records(recs, keep=kept, labels=labels)
        result[variant] = m
        result[variant + "_gate"] = ev.gate_check(base, m)
        result[variant + "_kept"] = kept

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{video}_tracks.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "shark_associated", "n_obs", "median", "q25", "frac_ge_095",
                        "keep_median_clip", "keep_q25_clip"])
            for r in rows:
                w.writerow([r["track_id"], r["shark"], r["n_obs"], round(r["median"], 4), round(r["q25"], 4),
                            round(r["frac_ge_095"], 4), r["keep_median_clip"], r["keep_q25_clip"]])
        with open(os.path.join(out_dir, f"{video}_summary.json"), "w") as f:
            json.dump({k: v for k, v in result.items() if k != "tracks"}, f, indent=2, default=str)
    return result


def _rank(x) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation (nan if either side is constant or there are < 3 points)."""
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(pos, neg) -> float:
    """P(random positive scores higher than random negative), ties count half (nan if either side is empty)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def probe_crops(video: str, root: str, scorers: Dict[str, Callable], max_obs: int = tv.MAX_OBS) -> list:
    """Diagnostic only (no tuning): for every track, score the same sampled crops with each scorer in
    `scorers` ({"raw": ..., "pad": ...}) and record crop size and aspect ratio (long side / short side)."""
    recs = ev.build_records(video, gt_dir(root, video), flair_dir(root, video))
    labels = ev.label_tracks(recs)
    imgs = img_dir(root, video)
    rows = []
    for tid in sorted(labels):
        frames = [i for i, r in enumerate(recs) if r.pred is not None and tid in r.pred and r.pred[tid].sum() > 0]
        for i in tv.sample_frames(frames, max_obs):
            path = ev.find_image(imgs, video, recs[i].gt_frame)
            crop = tv.crop_for_mask(np.array(Image.open(path).convert("RGB")), recs[i].pred[tid])
            if crop is None or not crop.size:
                continue
            h, w = crop.shape[:2]
            row = {"video": video, "track_id": tid, "shark": bool(labels[tid]["shark"]),
                   "frame": recs[i].gt_frame, "h": h, "w": w, "aspect": max(h, w) / min(h, w)}
            for name, sc in scorers.items():
                row["p_" + name] = float(sc([crop])[0])
            rows.append(row)
    return rows


def print_probe(rows: list, names=("raw", "pad")):
    """Per-track medians for each scorer, plus Spearman(aspect, p) per scorer (all crops, and shark tracks only)."""
    print(f"{'video':8s} {'track':>5} {'shark':>6} {'n':>3} {'med_aspect':>10} " + " ".join(f"{'med_' + n:>9}" for n in names))
    for key in sorted({(r["video"], r["track_id"]) for r in rows}):
        sub = [r for r in rows if (r["video"], r["track_id"]) == key]
        print(f"{key[0]:8s} {key[1]:>5} {str(sub[0]['shark']):>6} {len(sub):>3} "
              f"{np.median([r['aspect'] for r in sub]):>10.2f} "
              + " ".join(f"{np.median([r['p_' + n] for r in sub]):>9.3f}" for n in names))
    print("\nSeparability, shark-associated vs extra (AUC: 1.0 = perfect, 0.5 = chance, <0.5 = inverted):")
    for video in sorted({r["video"] for r in rows}):
        vr = [r for r in rows if r["video"] == video]
        if not any(r["shark"] for r in vr) or all(r["shark"] for r in vr):
            print(f"  {video}: needs both shark and extra tracks, skipped")
            continue
        parts = []
        for n in names:
            crop_auc = auc([r["p_" + n] for r in vr if r["shark"]], [r["p_" + n] for r in vr if not r["shark"]])
            meds = {}
            for r in vr:
                meds.setdefault((r["track_id"], r["shark"]), []).append(r["p_" + n])
            tm = {k: float(np.median(v)) for k, v in meds.items()}
            track_auc = auc([v for (t, s), v in tm.items() if s], [v for (t, s), v in tm.items() if not s])
            parts.append(f"{n}: crops {crop_auc:.2f}, track medians {track_auc:.2f}")
        print(f"  {video}: " + " | ".join(parts))
    print()
    for n in names:
        allc = spearman([r["aspect"] for r in rows], [r["p_" + n] for r in rows])
        sh = [r for r in rows if r["shark"]]
        shc = spearman([r["aspect"] for r in sh], [r["p_" + n] for r in sh])
        print(f"Spearman(aspect, P[{n}]): all crops {allc:+.2f} | shark tracks only {shc:+.2f}")
    print("(aspect = long side / short side, so a NEGATIVE correlation means narrower crops score lower)")


def crop_montage(video: str, root: str, track_ids, scorer: Callable, n: int = 6, max_obs: int = tv.MAX_OBS,
                 out_path: Optional[str] = None, cell: int = 3):
    """Diagnostic only (no tuning): show the exact CLIP crops for chosen tracks, with each crop's shark
    probability, so a low-scoring real shark can be inspected by eye. One row per track, n crops per row."""
    import matplotlib.pyplot as plt

    recs = ev.build_records(video, gt_dir(root, video), flair_dir(root, video))
    imgs = img_dir(root, video)
    fig, axes = plt.subplots(len(track_ids), n, figsize=(cell * n, cell * len(track_ids)), squeeze=False)
    for row, tid in enumerate(track_ids):
        frames = [i for i, r in enumerate(recs) if r.pred is not None and tid in r.pred and r.pred[tid].sum() > 0]
        chosen = tv.sample_frames(frames, max_obs)
        pick = [chosen[j] for j in np.unique(np.round(np.linspace(0, len(chosen) - 1, min(n, len(chosen)))).astype(int))] \
            if chosen else []
        for col in range(n):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(pick):
                continue
            i = pick[col]
            path = ev.find_image(imgs, video, recs[i].gt_frame)
            crop = tv.crop_for_mask(np.array(Image.open(path).convert("RGB")), recs[i].pred[tid])
            p = scorer([crop])[0]
            ax.imshow(crop)
            ax.set_title(f"track {tid} | frame {recs[i].gt_frame}\nP(shark)={p:.3f}", fontsize=9)
    fig.suptitle(f"{video}: CLIP crops (mask box + 25 px)", fontsize=12)
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=110)
    return fig


def print_summary(res: dict):
    v = res["video"]
    print(f"\n===== {v} =====")
    print("per-track CLIP (shark = overlaps GT shark on annotated frames):")
    print(f"  {'id':>4} {'shark':>6} {'n_obs':>5} {'median':>7} {'q25':>7} {'frac>=.95':>9}  keep_med keep_q25")
    for r in res["tracks"]:
        print(f"  {r['track_id']:>4} {str(r['shark']):>6} {r['n_obs']:>5} {r['median']:>7.3f} {r['q25']:>7.3f} "
              f"{r['frac_ge_095']:>9.2f}  {str(r['keep_median_clip']):>8} {str(r['keep_q25_clip']):>8}")
    cols = ["authors_dice", "union_dice", "recall", "tracks_retained", "shark_tracks_retained",
            "extra_tracks_retained"]
    print(f"\n  {'method':14s}" + "".join(f"{c:>23s}" for c in cols))
    for name in ("baseline", "oracle", "median_clip", "q25_clip"):
        m = res[name]
        print(f"  {name:14s}" + "".join(f"{m[c]:>23.4f}" if isinstance(m[c], float) else f"{m[c]:>23}" for c in cols))
    for variant in tv.VARIANTS:
        g = res[variant + "_gate"]
        print(f"  gate[{variant}]: extra removed={g['extra_removed_frac']:.0%} "
              f"recall_ok={g['ok_recall']} dice_ok={g['ok_dice']} -> {'PASS' if g['passed'] else 'NOT PASSED'}")
