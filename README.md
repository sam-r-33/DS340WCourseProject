# FLAIR reproduction and analysis (DS340W midterm)

This repository reproduces **FLAIR**, a zero-shot pipeline for tracking sharks in aerial video, and adds instrumentation to study where the pipeline spends its compute and how reliable its automatic filtering is.

- Course project: DS340W, Penn State
- Team: Sam Ruch & Shawn Agaon
- Original paper: Lalgudi et al. (2025), *Zero-shot shark tracking and biometrics from aerial imagery*, Methods in Ecology and Evolution (arXiv:2501.05717)
- Original code (MIT): https://github.com/conservation-technology-group/FLAIR

## What FLAIR does

1. SAM2 automatic mask generation on every Nth frame (here N = 30), then a size filter.
2. CLIP zero-shot filtering keeps candidate boxes that look like the target animal.
3. SAM2 video propagation tracks each accepted box through the video. A track is confirmed only when a later, independent detection overlaps it (IoU > 0.7); unconfirmed tracks are deleted.
4. An optional manual step keeps the tracks that are truly sharks.

## Notebooks

Run **`03_FLAIR_efficient_analysis.ipynb`**. The other two are kept to show how the work developed.

| Notebook | What it is |
|---|---|
| `notebooks/01_FLAIR_baseline.ipynb` | Our working copy of the original FLAIR Colab notebook, adapted to run from Google Drive. |
| `notebooks/02_FLAIR_efficient.ipynb` | Cleanup of 01: dead code removed, all paths and settings in one config cell, early error checks, results saved to Drive, built-in regression check. The algorithm and parameters are unchanged. |
| `notebooks/03_FLAIR_efficient_analysis.ipynb` | 02 plus read-only instrumentation: work counters, a snapshot of every track before FLAIR's final filter, a per-track feature table, and a comparison of CLIP scores for candidates on and off the kept track. Pipeline masks are unchanged. |

## Repository layout

- `notebooks/` : the three notebooks described above
- `results/` : outputs from the blacktip run (notebook 03)
  - `analysis_run_output.pdf`: printed Colab output of the full notebook 03 run, including the review plots and tables
  - `funnel.json`: stage counts and compute (frame-passes)
  - `track_features.csv`: per-track features for all 8 tracks FLAIR created
  - `clip_candidates.csv`: CLIP scores for all 22 candidate boxes
  - `candidate_summary.json`: summary of the on-track vs. off-track comparison
  - `clip_candidates.png`: plot of those scores
  - `run_info.json`: configuration, timings and environment

## How to run (Colab)

1. Open `03_FLAIR_efficient_analysis.ipynb` in Google Colab and choose a **GPU** runtime.
2. Put the input video on Google Drive and edit the **Configuration** cell: `PROJECT_DIR`, `SOURCE_VIDEO`, and optionally `RUN_NAME`, `CLIP_START` / `CLIP_END`.
3. Run all. The first run installs dependencies and downloads the SAM2 checkpoint (a few minutes).
4. Check the review grid, set `KEEP_KEYS` to the track ID(s) that are true sharks, and run again if needed.

Results are saved to `PROJECT_DIR/runs/<RUN_NAME>/` on Drive: zipped masks, boxes, the tracked video, `run_info.json`, and an `analysis/` folder.

## Data

We use the public dataset released with the FLAIR paper:

- Zenodo record: https://doi.org/10.5281/zenodo.15845961 (DOI 10.5281/zenodo.15845961, CC BY 4.0)
- FLAIR code archive: https://doi.org/10.5281/zenodo.16056059 (DOI 10.5281/zenodo.16056060, MIT)

The dataset is about 45 GB, so it is **not** stored in this repository. For the results below we used the example blacktip reef shark clip (`example_video_blacktip.mp4`, from `FLAIR_code.zip` in the Zenodo record and in the original FLAIR repository), first 8 seconds, 237 frames.

## Results so far (one clip)

All numbers come from a single 8-second blacktip shark clip and are descriptive, not a validated evaluation. Raw outputs are in `results/`.

**Reproduction check.** The cleaned pipeline's pruned masks matched the original notebook's pruned masks on all 237 frames (regression check: PASS, threshold mean Dice 0.95).

**Pipeline funnel**

| Stage | Count |
|---|---|
| Frames used for mask generation | 8 of 237 |
| SAM2 masks (raw) | 435 |
| After size filter (75 to 300 px) | 22 |
| Accepted by CLIP (confidence > 0.95) | 13 |
| Tracks created | 8 |
| Confirmed by FLAIR | 1 |
| Deleted by FLAIR's final filter | 7 |

**Compute.** Propagation processed 986 frames in total, 4.16 times the 237 video frames. 747 of those frame-passes (76%) were spent in calls launched for tracks that FLAIR later deleted; 239 (24%) went to the surviving shark track. Stage times: mask generation 47 s, CLIP 7 s, propagation 171 s. Note that "deleted" does not always mean "false"; see below.

**CLIP scores.** We labeled each CLIP candidate by whether it overlaps the track kept in `KEEP_KEYS` (7 candidates on the kept track, 15 not). The weakest on-track probability was 0.992 and two off-track candidates scored above it; a logit margin (target prompt minus the strongest other prompt) was no better (three off-track candidates above the weakest on-track value). So a single-frame CLIP score did not cleanly separate shark from non-shark candidates on this clip.

**Observation to confirm.** Two deleted tracks (tracks 1 and 6) appear to overlap the kept shark; they probably started as separate tracks because their candidate boxes overlapped the existing shark track below FLAIR's 0.7 duplicate threshold. We are verifying this with a per-frame mask-overlap check.

## Limitations

- One clip, one animal. Nothing here shows that any filter rule generalizes.
- Candidate labels come from our own `KEEP_KEYS` choice.
- Tracks that overlap the shark but were deleted count as "deleted" in the compute figures, not as false detections.

## Planned work

- Evaluate whether track-level evidence (how long a mask persists within its propagation window, CLIP scores across a track's frames, mask geometry) can replace the manual pruning step, on multiple videos with held-out evaluation.
- Test stopping propagation early for tracks whose mask disappears, and whether FLAIR's confirmation rule can be made less brittle, while keeping a background search so newly entering sharks are not missed.

## Related work that motivated this

- Lalgudi et al. (2025), FLAIR, Methods in Ecology and Evolution.
- DiGiacomo et al. (2026), PLOS ONE, white shark aerial morphometrics (altitude-aware cropping and mask-quality ideas).
- Delgado-Garcia et al. (2026), Journal of Fish Biology, whale shark detection (aggregating predictions over several frames to reduce false positives).

## License and credits

This repository's code is released under the MIT License (see `LICENSE`). It builds on the MIT-licensed FLAIR code by Conservation Technology Group; the dataset is CC BY 4.0 and credited above. SAM2 and CLIP are used under their own licenses.
