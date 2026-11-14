Offline CLIP gate on the released FLAIR masks: Colab usage (Drive mounted).

Put the .py files from this folder in Drive at DS340WCourseProject/code/. Nothing here modifies any data;
results are written to the output folders named below. Needs UnzippedData/ (ground truth + released masks) and
diagnostics/dice_summary.csv in DS340WCourseProject/.

Tests (synthetic data, no GPU or data needed):  python3 test_modules.py ; python3 test_gate_runner.py

# --- cell 1: setup + self-test of the scorer (CPU, no CLIP needed) ---
import sys; sys.path.insert(0, "/content/drive/MyDrive/DS340WCourseProject/code")
import importlib, evaluation, track_validation, gate_runner
for m in (evaluation, track_validation, gate_runner): importlib.reload(m)
import gate_runner as gr

PROJECT = "/content/drive/MyDrive/DS340WCourseProject"
ROOT    = f"{PROJECT}/UnzippedData"
SUMMARY = f"{PROJECT}/diagnostics/dice_summary.csv"   # edit to wherever dice_summary.csv is
computed, problems = gr.self_test(ROOT, SUMMARY, ["video5", "video3"])   # start small; add video2, video6 after
# problems == [] means the scorer reproduces the released scores in dice_summary.csv

# --- cell 2: the pre-specified CLIP gate (frozen FLAIR preprocessing; GPU runtime recommended) ---
import track_validation as tv
raw = tv.make_clip_scorer()
OUT = f"{PROJECT}/gate_results"
results = {}
for v in ["video5", "video2", "video6", "video3"]:      # order from the protocol; video3 = clean control
    results[v] = gr.run_video(v, ROOT, raw, out_dir=OUT)
    gr.print_summary(results[v])

# --- cell 3 (optional): no-harm check on the remaining clips ---
for v in ["video1", "video4", "video7", "white1", "blacktip"]:
    gr.print_summary(gr.run_video(v, ROOT, raw, out_dir=OUT))

# --- cell 4 (optional, EXPLORATORY, post hoc): crops padded to square before CLIP ---
# Chosen after seeing videos 2, 3, 5 and 6, so it is not part of the pre-specified protocol.
pad = tv.make_clip_scorer(pad_square=True)
for v in ["video5", "video2", "video6", "video3", "video1", "video4", "video7", "white1", "blacktip"]:
    gr.print_summary(gr.run_video(v, ROOT, pad, out_dir=f"{PROJECT}/gate_results_pad"))

Notes
- Output per video: {video}_tracks.csv (per-track CLIP median / q25) and {video}_summary.json (metrics for the
  baseline, oracle, median_clip and q25_clip, with the gate result).
- white1 and blacktip have GT masks for frames without an image; tracks are scored only on frames that have one
  (a note is printed with the number skipped).
- End-to-end runs of our own pipeline use notebook 04_FLAIR_track_validation.ipynb, not this script.
