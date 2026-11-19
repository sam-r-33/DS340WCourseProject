"""Offline replay of candidate-admission rules on the logged FLAIR run (blacktip clip, 237 frames).

WHY A REPLAY: FLAIR first scores every candidate box on 8 sampled frames (SAM 2 mask generation + CLIP), and only
then starts SAM 2 video propagation. So the full candidate list is known before any propagation cost is paid.
This script takes the logged candidate table (results/clip_candidates.csv) and asks: if an extra admission rule
had been applied right before propagation, which tracks would have been launched and how many SAM 2 frame-passes
would that have cost?

The cost model reproduces FLAIR's own propagation schedule (3-s window, every 30th frame) and is CHECKED against
the logged baseline (868 frame-passes in new-track calls = 121 + 747, and 118 in re-propagation, total 986).
It is a simulation on logged data, not a new SAM 2 run: it does not recompute any mask.

Usage:  python analysis/replay_gate_comparison.py [results_dir]
"""
import csv, json, math, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "gate_replay")
os.makedirs(OUT, exist_ok=True)

N_VIDEO, FRAME_GAP, WINDOW = 237, 30, 3          # clip length, sampled-frame spacing, propagation window (s)
N_SAMPLED = 8
CLIP_CONF = 0.95
DUP_IOU = 0.9   # a candidate that matches the kept shark track at least this well is a duplicate (FLAIR: IoU > 0.7)

# ---------------------------------------------------------------- load logged candidates
cands = []
with open(os.path.join(RES, "clip_candidates.csv")) as f:
    for r in csv.DictReader(f):
        x1, y1, x2, y2 = json.loads(r["bbox"])
        cands.append({"f": int(r["frame_sampled"]), "j": int(r["box_idx"]), "iou_kept": float(r["iou_with_kept"]),
                      "p": float(r["p_target"]), "m": float(r["margin"]), "accepted": r["accepted"] == "True",
                      "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "diag": math.hypot(x2 - x1, y2 - y1)})
cands.sort(key=lambda c: (c["f"], c["j"]))
tracks_logged = []
with open(os.path.join(RES, "track_features.csv")) as f:
    for r in csv.DictReader(f):
        tracks_logged.append((int(r["track_id"]), int(r["created_at_sampled_frame"]), r["status"]))


# ---------------------------------------------------------------- admission rules (all act on CLIP-accepted candidates)
def stage1(c, tau, margin_min):
    return c["p"] > tau and c["accepted"] and c["m"] >= margin_min


def corroborated(c, nodes, lam):
    """True if an admitted candidate in an ADJACENT sampled frame lies within lam x mean box diagonal."""
    for o in nodes:
        if abs(o["f"] - c["f"]) == 1:
            d = math.hypot(o["cx"] - c["cx"], o["cy"] - c["cy"])
            if d <= lam * (o["diag"] + c["diag"]) / 2:
                return True
    return False


def admit(tau=CLIP_CONF, margin_min=-1e9, lam=None, exempt_last=False):
    nodes = [c for c in cands if stage1(c, tau, margin_min)]
    if lam is None:
        return nodes
    out = []
    for c in nodes:
        if (exempt_last and c["f"] == N_SAMPLED - 1) or corroborated(c, nodes, lam):
            out.append(c)
    return out


# ---------------------------------------------------------------- FLAIR propagation cost model
def call_frames(a, reprop=False):
    """SAM 2 frames processed by one propagate call made while handling sampled frame a (FLAIR's own schedule)."""
    if reprop:
        start, max_frame = FRAME_GAP * a, FRAME_GAP * WINDOW
    else:
        start = max(0, FRAME_GAP * (a - WINDOW))
        max_frame = FRAME_GAP * WINDOW + (FRAME_GAP * a - start)
    return max(0, min(max_frame + 1, N_VIDEO - start))


def replay(admitted):
    """Which tracks FLAIR would create from these candidates, and what propagation costs."""
    new_tracks, shark_track = [], None            # shark_track = (created_f, confirmed_f or None)
    confirm_f = None
    for c in admitted:                            # already in (frame, box) order
        is_dup = c["iou_kept"] >= DUP_IOU and c["f"] >= 2 and shark_track is not None
        if is_dup:
            if confirm_f is None and c["f"] != shark_track:
                confirm_f = c["f"]
            continue
        new_tracks.append(c)
        if c["iou_kept"] >= DUP_IOU and shark_track is None:
            shark_track = c["f"]                  # first well-matching candidate starts the kept-shark track
    calls = sorted({c["f"] for c in new_tracks})
    new_passes = sum(call_frames(a) for a in calls)
    reprop_passes, r = 0, None
    if shark_track is not None and confirm_f is not None:
        r = shark_track + WINDOW
        while r < N_SAMPLED:
            if confirm_f <= r:
                reprop_passes += call_frames(r, reprop=True)
                r += WINDOW
            else:
                break
    off_shark = [c for c in new_tracks if c["iou_kept"] == 0.0]
    overlap = [c for c in new_tracks if c["iou_kept"] > 0.0]
    return {"tracks_launched": len(new_tracks), "tracks_off_shark": len(off_shark), "tracks_overlapping_shark": len(overlap),
            "calls": calls, "new_call_passes": new_passes, "reprop_passes": reprop_passes,
            "total_passes": new_passes + reprop_passes, "shark_confirmed": confirm_f is not None, "confirm_frame": confirm_f,
            "shark_candidates_kept": sum(1 for c in admitted if c["iou_kept"] > 0.0),
            "candidates_admitted": len(admitted)}


# ---------------------------------------------------------------- validation against the logged baseline run
base = replay(admit())
per_frame = [sum(1 for c in cands if c["f"] == f and c in [] ) for f in range(N_SAMPLED)]
logged_created = sorted(t[1] for t in tracks_logged)
replayed_created = sorted(c["f"] for c in
                          [c for c in admit() if not (c["iou_kept"] >= DUP_IOU and c["f"] >= 2)])
assert replayed_created == logged_created, (replayed_created, logged_created)
assert base["new_call_passes"] == 868, base            # 121 (kept track call) + 747 (all other calls), from funnel.json
assert base["reprop_passes"] == 118, base               # funnel.json: frame_passes_repropagation
assert base["total_passes"] == 986, base                # funnel.json: propagation_frame_passes
print("Replay reproduces the logged baseline exactly: 8 tracks, 868 new-call + 118 re-propagation = 986 frame-passes.")

# ---------------------------------------------------------------- variants compared in the paper
variants = [
    ("A", "FLAIR baseline (CLIP p > 0.95)", dict()),
    ("B1", "Stricter CLIP threshold, p > 0.97", dict(tau=0.97)),
    ("B2", "Stricter CLIP threshold, p > 0.99", dict(tau=0.99)),
    ("B3", "Stricter CLIP threshold, p > 0.995", dict(tau=0.995)),
    ("C1", "Logit-margin filter, margin >= 4", dict(margin_min=4.0)),
    ("C2", "Logit-margin filter, margin >= 5", dict(margin_min=5.0)),
    ("D1", "Temporal corroboration (TCC), lambda = 1", dict(lam=1.0)),
    ("D2", "TCC, lambda = 1, last sampled frame exempt", dict(lam=1.0, exempt_last=True)),
    ("D3", "TCC (lambda = 1) + margin >= 4", dict(lam=1.0, margin_min=4.0)),
]
rows = []
for vid, name, kw in variants:
    r = replay(admit(**kw)); r.update({"id": vid, "name": name})
    r["passes_saved_pct"] = round(100 * (1 - r["total_passes"] / base["total_passes"]), 1)
    rows.append(r)

# ---------------------------------------------------------------- sensitivity sweeps
def sweep(param, values, **fixed):
    out = []
    for v in values:
        kw = dict(fixed); kw[param] = v
        r = replay(admit(**kw)); r["value"] = v
        r["passes_saved_pct"] = round(100 * (1 - r["total_passes"] / base["total_passes"]), 1)
        out.append(r)
    return out

lam_vals = [0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
tau_vals = [0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.992, 0.994, 0.995, 0.997, 0.999]
mar_vals = [0, 2, 3, 4, 5, 5.5, 6, 6.5, 7]
sw_lam, sw_tau, sw_mar = sweep("lam", lam_vals), sweep("tau", tau_vals), sweep("margin_min", mar_vals)

# robustness of the TCC rule: normalised distances actually seen between accepted candidates in adjacent frames
acc = admit()
def ndist(a, b): return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / ((a["diag"] + b["diag"]) / 2)
shark_links, other_nearest = [], []
for c in acc:
    ds = [ndist(c, o) for o in acc if abs(o["f"] - c["f"]) == 1]
    if c["iou_kept"] > 0:
        # the shark chain only needs ONE link to a neighbouring shark candidate
        dsh = [ndist(c, o) for o in acc if abs(o["f"] - c["f"]) == 1 and o["iou_kept"] > 0]
        if dsh: shark_links.append(min(dsh))
    else:
        other_nearest.append(min(ds) if ds else float("inf"))
robust = {"max_over_shark_candidates_of_nearest_shark_neighbour": round(max(shark_links), 3),
          "min_over_off_shark_candidates_of_nearest_neighbour": round(min(other_nearest), 3)}

# calls and costs for the record
calls_cost = {a: call_frames(a) for a in range(N_SAMPLED)}
summary = {"baseline_validation": {"new_call_passes": 868, "reprop_passes": 118, "total_passes": 986, "tracks": 8},
           "call_cost_by_sampled_frame": calls_cost, "variants": rows, "sweep_lambda": sw_lam, "sweep_tau": sw_tau,
           "sweep_margin": sw_mar, "robustness": robust,
           "accepted_candidates": [{k: c[k] for k in ("f", "j", "p", "m", "iou_kept", "cx", "cy", "diag")} for c in acc]}
json.dump(summary, open(os.path.join(OUT, "gate_replay_summary.json"), "w"), indent=2, default=str)
cols = ["id", "name", "candidates_admitted", "tracks_launched", "tracks_off_shark", "tracks_overlapping_shark",
        "shark_confirmed", "shark_candidates_kept", "new_call_passes", "reprop_passes", "total_passes", "passes_saved_pct"]
with open(os.path.join(OUT, "variant_comparison.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

print(f"{'id':<3} {'variant':<48} {'trk':>3} {'off':>3} {'ovl':>3} {'conf':>4} {'sharkC':>6} {'passes':>6} {'saved%':>6}")
for r in rows:
    print(f"{r['id']:<3} {r['name']:<48} {r['tracks_launched']:>3} {r['tracks_off_shark']:>3} {r['tracks_overlapping_shark']:>3} "
          f"{str(r['shark_confirmed']):>4} {r['shark_candidates_kept']:>6} {r['total_passes']:>6} {r['passes_saved_pct']:>6}")
print("\nTCC robustness (normalised centre distance, units of mean box diagonal):", robust)

# ---------------------------------------------------------------- operating windows (post-hoc, uses the on/off-shark labels)
def window(param, grid, **fixed):
    """Range of the parameter for which the kept shark is confirmed as early as in the baseline AND no off-shark track is launched."""
    ok = []
    for v in grid:
        kw = dict(fixed); kw[param] = v
        r = replay(admit(**kw))
        # "safe" = no off-shark track launched AND the kept shark is confirmed no later than in the baseline run
        if r["shark_confirmed"] and r["confirm_frame"] <= base["confirm_frame"] and r["tracks_off_shark"] == 0:
            ok.append(v)
    return (round(min(ok), 4), round(max(ok), 4)) if ok else None

win = {
    "lambda": window("lam", [x / 20 for x in range(1, 161)]),
    "clip_threshold": window("tau", [0.95 + k * 0.0001 for k in range(0, 500)]),
    "logit_margin": window("margin_min", [x / 20 for x in range(0, 161)]),
}
summary["operating_windows"] = win
json.dump(summary, open(os.path.join(OUT, "gate_replay_summary.json"), "w"), indent=2, default=str)
print("Operating windows (no off-shark track AND shark confirmed no later than baseline):", win)
