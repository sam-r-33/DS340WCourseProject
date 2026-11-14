# Evaluation Protocol: automatic track-level CLIP validation

DS340W Group 7 (Sam Ruch, Shawn Agaon). Reproduction of FLAIR (Lalgudi et al. 2025).

## Purpose of this file

This file records, in one place, the rule we committed to *before* looking at
results, how we score it, and what we count as success. It exists so that the
numbers in the paper can be checked against a rule that was fixed in advance,
and so that anything we tried afterwards is clearly labeled as exploratory.

## The modified block

FLAIR removes false tracks by hand: after propagation, a person inspects each
track and lists the ones to keep (`KEEP_KEYS`). We replace that manual step with
an automatic, track-level CLIP check.

## Frozen rule (not tuned on any clip)

- Sample up to 10 evenly spaced frames per track.
- Crop each frame exactly as FLAIR does (mask bounding box + 25 px padding).
- Score the crop with FLAIR's own CLIP model, prompts and preprocessing.
- Aggregate per track: `median_clip` (primary) is the median of the frame
  scores; `q25_clip` (secondary) is the 25th percentile.
- Keep a track if its aggregate is >= 0.95.
- The 0.95 threshold is FLAIR's own candidate-filter value. It is fixed and is
  not adjusted to rescue any video.

## Exploratory variant (NOT pre-specified)

A pad-to-square variant (pad the crop to a square before CLIP's center crop) was
added after seeing that center-cropping trims elongated sharks. It was designed
after inspecting results, so it is reported only as post hoc and exploratory.

## Metrics

- **Best-match Dice** (the authors' metric): cannot penalize extra tracks and
  cannot be raised by pruning.
- **Union Dice** (ours): penalizes extra area from false tracks.
- **Recall**: a ground-truth shark counts as found if any kept mask has
  Dice >= 0.5 with it.
- **Shark-associated track**: at least 50% of the mask area lies on a GT shark,
  on at least 50% of annotated frames where the track has a mask.
- **Oracle**: keeps exactly the shark-associated tracks. It uses ground truth,
  so it is a reference ceiling, not deployable.

## Two kinds of test

1. **Offline gate** on the released masks (9 clips). Scores tracks on annotated
   frames only, where the shark is likely visible, so it is optimistic.
2. **End-to-end** run of notebook 04 on raw video. Scores each track on up to 10
   frames across the whole track. This is the deployable test and the one that
   counts.

## Pass criteria (gate)

- Extra tracks removed: at least 50%.
- Recall: not below the unfiltered baseline.
- Best-match Dice drop: at most 0.01.
- The primary gate requires `median_clip` to pass on videos 5 and 2.

## Rules we hold ourselves to

- Report failures. A harmed clip stays in the tables.
- Do not retune the threshold or aggregation to fix videos 5 or 6.
- Any rule chosen using these clips needs a separate no-harm check on clips not
  used to choose it.
- Results are relative to our own baseline run on each clip (our run produced
  fewer tracks than the released masks on video 2).

## Results so far

Offline, frozen scorer: **gate fails**. Video 2 and 6 pass; video 5 fails (shark
0.916 below two extras); video 3 is harmed (a real shark at 0.871 deleted);
white1's only shark is deleted by raw median (0.937).

End-to-end, frozen scorer: helped 1 of 4 clips (video 2, union 0.142 -> 0.686,
recall unchanged), harmed 2 (video 5, video 6: shark deleted, Dice -> 0),
neutral 1 (blacktip, 8 s clip).

Post hoc padded median: no real shark deleted on any of the 9 offline clips,
but only videos 1, 4 and 7 are clean no-harm evidence, and margins are thin.

## Reproduction

See `code/README_gate.txt` for the offline gate and `04_FLAIR_track_validation.ipynb`
for end-to-end runs. Tests: `python code/test_modules.py` and
`python code/test_gate_runner.py`.
