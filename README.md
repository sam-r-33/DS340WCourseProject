# FLAIR reproduction, analysis, and track validation (DS340W)

This repository reproduces **FLAIR**, a zero-shot pipeline for tracking sharks in aerial video, adds instrumentation to study where the pipeline spends computation and how its candidate and track filtering behaves, and tests one modified block: an automatic, track-level CLIP check that replaces FLAIR's manual track pruning.

- Course project: DS340W, Penn State
- Team: Sam Ruch & Shawn Agaon
- Original paper: Lalgudi et al. (2025), *Zero-shot shark tracking and biometrics from aerial imagery*, Methods in Ecology and Evolution (arXiv:2501.05717)
- Original code (MIT): https://github.com/conservation-technology-group/FLAIR

## What FLAIR does

1. SAM2 automatic mask generation on sampled frames (30-frame spacing in the blacktip reproduction; other clips use their configured sampling interval, e.g. 24 for the 23.976 fps videos 5 and 6), then a size filter.
2. CLIP zero-shot filtering keeps candidate boxes that look like the target animal.
3. SAM2 video propagation tracks each accepted box through the video. A track is confirmed only when a later, independent detection overlaps it (IoU > 0.7); unconfirmed tracks are deleted.
4. An optional manual step keeps the tracks that are truly sharks.

## Notebooks

**`02_FLAIR_efficient.ipynb`** is our reproduction of the parent paper's pipeline. **`03_FLAIR_efficient_analysis.ipynb`** builds on it for our own analysis. **`04_FLAIR_track_validation.ipynb`** is our modified pipeline (the novel block). `01` is the original-style baseline.

| Notebook | What it is |
|---|---|
| `notebooks/01_FLAIR_baseline.ipynb` | Our working copy of the original FLAIR Colab notebook, adapted to run from Google Drive. |
| `notebooks/02_FLAIR_efficient.ipynb` | Cleanup of 01: dead code removed, all paths and settings in one config cell, early error checks, results saved to Drive, built-in regression check. The core FLAIR algorithm and reproduction parameters are preserved; on the example clip, the cleaned notebook matches the baseline output (mean Dice 1.0). The notebook ends with a Parent Paper Reproduction Check section. |
| `notebooks/03_FLAIR_efficient_analysis.ipynb` | 02 plus read-only instrumentation: work counters, a snapshot of every track before FLAIR's final filter, a per-track feature table, and a comparison of CLIP scores for candidates on and off the kept track. Pipeline masks are unchanged. |
| `notebooks/04_FLAIR_track_validation.ipynb` | A copy of 02 (without 03's instrumentation) with the manual `KEEP_KEYS` pruning step replaced by automatic track-level CLIP validation (set `TRACK_FILTER` to `"manual"`, `"median_clip"` or `"q25_clip"`). Adds an optional ground-truth evaluation cell and writes `validation/tracks.csv`, `selection.json` and `evaluation.json` for each run. Everything else in the pipeline is unchanged. |

## Repository layout

- `notebooks/` : the four notebooks described above
- `code/` : the modified block and its offline test harness (see "Novel block" below)
  - `track_validation.py`: the track-level CLIP validation block
  - `evaluation.py`: metrics (best-match Dice, union Dice, recall), track labeling, gate check
  - `gate_runner.py`: offline gate on the released masks, plus diagnostics
  - `test_modules.py`, `test_gate_runner.py`: tests on synthetic data (run with `python3 <file>`)
  - `README_gate.txt`: how to run the offline gate in Colab
- `PROTOCOL.md` : the frozen rule, metrics and pass criteria, written before the multi-clip results
- `results/track_validation/` : result files for the novel block
  - `offline_gate/`: frozen scorer on released masks, 9 clips (`diagnostics/` has crop montages for videos 3 and 5)
  - `offline_gate_padded/`: post hoc, exploratory pad-to-square variant, 9 clips
  - `e2e_blacktip/`, `e2e_video2/`, `e2e_video5/`, `e2e_video6/`: end-to-end runs of notebook 04 (`validation/` holds per-track scores, the selection and the evaluation; `run_info.json` holds the settings)
- `results/` : outputs from the blacktip run (notebook 03), plus `track_validation/` described above
  - `analysis_run_output.pdf`: printed Colab output of the full notebook 03 run, including the review plots and tables
  - `funnel.json`: stage counts and compute (frame-passes)
  - `track_features.csv`: per-track features for all 8 tracks FLAIR created
  - `clip_candidates.csv`: CLIP scores for all 22 candidate boxes
  - `candidate_summary.json`: summary of the on-track vs. off-track comparison
  - `clip_candidates.png`: plot of those scores
  - `run_info.json`: configuration, timings and environment

## How to run (Colab)

1. Open `02_FLAIR_efficient.ipynb` (the reproduction), `03_FLAIR_efficient_analysis.ipynb` (the analysis) or `04_FLAIR_track_validation.ipynb` (the modified pipeline) in Google Colab and choose a **GPU** runtime.
2. Put the input video on Google Drive and edit the **Configuration** cell: `PROJECT_DIR`, `SOURCE_VIDEO`, and optionally `RUN_NAME`, `CLIP_START` / `CLIP_END`.
3. Run all. The first run installs dependencies and downloads the SAM2 checkpoint if it is not already on Drive (a few minutes).
4. For notebooks 02 and 03, or notebook 04 with `TRACK_FILTER = "manual"`, check the review grid, set `KEEP_KEYS` to the track ID(s) that are true sharks, and run again if needed. In notebook 04, `KEEP_KEYS` applies only to the manual path; with `TRACK_FILTER = "median_clip"` (the default) or `"q25_clip"` the tracks are selected automatically.

Results are saved to `PROJECT_DIR/runs/<RUN_NAME>/` on Drive: zipped masks, boxes, the tracked video, `run_info.json`, an `analysis/` folder (notebook 03 only) and a `validation/` folder (notebook 04 only).

## Data

We use the public dataset released with the FLAIR paper:

- Zenodo record: https://doi.org/10.5281/zenodo.15845961 (DOI 10.5281/zenodo.15845961, CC BY 4.0)
- FLAIR code archive: https://doi.org/10.5281/zenodo.16056059 (DOI 10.5281/zenodo.16056059, MIT)

The dataset is about 45 GB, so it is **not** stored in this repository. The reproduction and instrumentation study uses the example blacktip reef shark clip (`example_video_blacktip.mp4`, from `FLAIR_code.zip` in the Zenodo record and in the original FLAIR repository), first 8 seconds, 237 frames. The track-validation experiments additionally use the released masks and ground-truth sampled frames for nine clips, plus end-to-end runs on blacktip and videos 2, 5 and 6.

## Results: reproduction and instrumentation (one clip)

All numbers in this section come from a single 8-second blacktip shark clip and are descriptive, not a validated evaluation. Raw outputs are in `results/` (top level).

**Reproduction check.** Notebook 02 on the first 8 seconds of the example blacktip clip (237 frames, NVIDIA L4) produced pruned masks matching the original notebook's output (mean and minimum Dice 1.0; regression check: PASS, threshold mean Dice 0.95). **This compares against the original notebook's output, not the paper's ground-truth Dice.**

**Pipeline funnel**

| Stage | Count |
|---|---|
| Frames used for mask generation | 8 of 237 |
| SAM2 masks (raw) | 435 |
| After size filter (bounding-box dimensions within the configured 75 to 300 px range) | 22 |
| Accepted by CLIP (confidence > 0.95) | 13 |
| Tracks created | 8 |
| Confirmed by FLAIR | 1 |
| Deleted by FLAIR's final filter | 7 |

**Compute.** Propagation processed 986 frames in total, 4.16 times the 237 video frames. 747 of those frame-passes (76%) were spent in calls launched for tracks that FLAIR later deleted; 239 (24%) went to the surviving shark track. Stage times: mask generation 47 s, CLIP 7 s, propagation 171 s. Note that "deleted" does not always mean "false"; see below.

**CLIP scores.** We labeled each CLIP candidate by whether it overlaps the track kept in `KEEP_KEYS` (7 candidates on the kept track, 15 not). The weakest on-track probability was 0.992 and two off-track candidates scored above it; a logit margin (target prompt minus the strongest other prompt) was no better (three off-track candidates above the weakest on-track value). So a single-frame CLIP score did not cleanly separate shark from non-shark candidates on this clip.

## Novel block: automatic track-level CLIP validation

**What changed.** FLAIR ends with a manual step: a person looks at each track and lists the real sharks in `KEEP_KEYS`. In notebook 04 that step is replaced by a rule that uses FLAIR's own CLIP model. For each track we sample up to 10 evenly spaced frames, crop each frame the way FLAIR does (mask box + 25 px), score the crop with the same CLIP model, prompts and preprocessing, and keep the track if the median score (`median_clip`, primary) or the 25th percentile (`q25_clip`, secondary) is at least 0.95. The threshold is FLAIR's own candidate-filter value and was fixed in advance; it was not tuned. The rule, metrics and pass criteria are in [`PROTOCOL.md`](PROTOCOL.md).

**Metrics.** *Best-match Dice* is the parent paper's metric; it cannot penalize extra tracks. *Union Dice* is ours and does penalize extra area. *Recall* counts a ground-truth shark as found if any kept mask has Dice at least 0.5 with it. The *oracle* keeps exactly the tracks that lie on a ground-truth shark; it uses ground truth and is not deployable, so it is only a reference.

### End-to-end evaluation on source video (notebook 04)

Each track is scored on up to 10 frames across the whole track. "Baseline" keeps every track FLAIR produced in the same run.

| Clip | Tracks (shark / extra) | Baseline Dice (best-match / union) | After `median_clip` | Outcome |
|---|---|---|---|---|
| blacktip (8 s) | 1 (1 / 0) | 0.903 / 0.903 | unchanged | neutral |
| video 2 | 15 (2 / 13) | 0.659 / 0.142, recall 0.934 | 0.659 / 0.686, recall 0.934, 12 of 13 extras removed | helped |
| video 5 | 3 (1 / 2) | 0.822 / 0.371, recall 1.0 | shark deleted: 0.000 / 0.000, recall 0 | harmed |
| video 6 | 3 (1 / 2) | 0.717 / 0.281, recall 0.767 | all tracks deleted: 0.000 / 0.000, recall 0 | harmed |

Helped on 1 of 4 clips, harmed on 2, neutral on 1. In video 2 one false track (track 13) is kept at 0.989. In videos 5 and 6 the real shark's median score (0.893 and 0.942) is below the extras' or below the threshold. Our end-to-end baseline on video 2 is lower than the released masks (our run produced 15 tracks vs. 27), so these numbers are relative to our own runs.

### Offline gate on the released masks (9 clips, frozen scorer)

Median-score rule, scored on annotated frames only. Details per clip are in `results/track_validation/offline_gate/`.

| Clip | Extra tracks removed | Real shark deleted? | Notes |
|---|---|---|---|
| video 2 | 23 of 24 | no | union Dice 0.124 to 0.674 (oracle 0.689) |
| video 6 | 2 of 3 | no | union 0.633 to 0.724 (oracle 0.788) |
| video 5 | 0 of 2 | **yes** | shark 0.916 below both extras (0.969, 0.956); Dice 0.814 to 0 |
| video 3 | none present | **yes** | a real shark at 0.871 deleted; Dice 0.852 to 0.585, recall 1.0 to 0.61 |
| white1 | none present | **yes** | only shark (0.937, 4 observations) deleted; Dice 0.920 to 0 |
| videos 1, 4, 7, blacktip | none present | no | no change (video 7's shark is close to the line at 0.955) |

**The pre-specified gate fails** (it required `median_clip` to pass on videos 5 and 2 and not to harm others). The raw 25th-percentile rule is worse: it deletes real sharks on videos 1 and 7 as well.

**The offline and end-to-end evaluations can differ.** The offline gate scores tracks only on annotated frames, where the shark is likely visible; end-to-end runs score each track across its whole life. On video 6, for example, the shark scores 0.974 on annotated frames (offline) but its end-to-end median is 0.942. We have not verified this explanation beyond that one diagnostic, so the offline gate may be optimistic, but we cannot say by how much.

### Exploratory: pad-to-square variant (post hoc)

CLIP's default preprocessing center-crops to a square, which can trim elongated sharks. After seeing the results above we tried padding each crop to a square first. **This was chosen after looking at videos 2, 3, 5 and 6, so it is not part of the pre-specified protocol and is reported only as exploratory** (`results/track_validation/offline_gate_padded/`). In the offline annotated-frame evaluation, the padded median rule did not delete any shark-associated track on the nine clips; video 6 had all extras removed (union Dice 0.788, equal to the oracle); video 2 had 19 of 24 extras removed (union 0.267); video 5 had no extras removed. The margins are thin (white1's shark scores 0.957, video 7's 0.967), the variant has not been tested end-to-end, and only videos 1, 4 and 7 are clean no-harm evidence, since the others were used to design or inspect it.

### Reproducing the novel-block results

- **Tests (no GPU or data needed):** `python3 code/test_modules.py` and `python3 code/test_gate_runner.py`.
- **Offline gate:** follow `code/README_gate.txt` in Colab (needs the released masks and ground truth from the Zenodo record, in `UnzippedData/`).
- **End-to-end:** open `notebooks/04_FLAIR_track_validation.ipynb`, set `TRACK_FILTER`, `SOURCE_VIDEO` (or `FRAMES_SOURCE_DIR`) and the ground-truth paths in the Configuration cell, and run all. Do not re-run the Configuration cell partway through a session, because it resets the run state.

## Limitations

- The reproduction and instrumentation results use one clip, one animal. Nothing there shows that any filter rule generalizes.
- The track-validation evidence is small: four end-to-end clips and nine offline clips, several with only one or two sharks. The offline gate may be optimistic, and the pad-to-square variant is post hoc and untested end-to-end.
- Candidate labels in the instrumentation results come from our own `KEEP_KEYS` choice. In the track-validation results, track labels (shark-associated or not) come from ground-truth masks.
- Our end-to-end baselines differ from the released masks (different track counts), so results are relative to our own runs.
- Tracks that overlap the shark but were deleted count as "deleted" in the compute figures, not as false detections.
- The reproduction check shows our cleaned notebook matches the original notebook's output. We have not yet reproduced the paper's reported Dice score against ground-truth masks.

## Planned work

- Develop richer track-level validation beyond a fixed CLIP threshold, including geometry, contrast, hybrid semantic/geometric features (how long a mask persists within its propagation window, mask and trajectory features) and alternative temporal aggregation, with held-out clips and a separate no-harm evaluation.
- Test early stopping after sustained mask disappearance or low confidence, and whether FLAIR's confirmation rule can be made less brittle, while keeping a background search so newly entering sharks are not missed.

## Related work that motivated this

- Lalgudi et al. (2025), FLAIR, Methods in Ecology and Evolution.
- DiGiacomo et al. (2026), PLOS ONE, white shark aerial morphometrics (altitude-aware cropping and mask-quality ideas).
- Delgado-Garcia et al. (2026), Journal of Fish Biology, whale shark detection (aggregating predictions over several frames to reduce false positives).

## License and credits

This repository's code is released under the MIT License (see `LICENSE`). It builds on the MIT-licensed FLAIR code by Conservation Technology Group; the dataset is CC BY 4.0 and credited above. SAM2 and CLIP are used under their own licenses.
