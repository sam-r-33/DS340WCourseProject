"""Test gate_runner.py against a miniature fake tree that mimics the real Zenodo folder layout.

Run:  python3 test_gate_runner.py
"""
import csv
import os
import tempfile

import numpy as np
from PIL import Image

import evaluation as ev
import gate_runner as gr

H, W = 200, 300


def rect(r0, r1, c0, c1):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def fake_scorer(crops):
    return [0.99 if c.max() > 200 else 0.3 for c in crops]


def build_tree(root):
    masks = os.path.join(root, gr.SAMPLED, "masks", gr.LAYOUT["video5"][0])
    images = os.path.join(root, gr.SAMPLED, "images", gr.LAYOUT["video5"][1])
    flair = os.path.join(root, "predicted_masks", "FLAIR", "video_segments_video5_small_3")
    hil = os.path.join(root, "predicted_masks", "HiL", "video_segments_HiL_video5_small")
    other = os.path.join(root, "predicted_masks", "FLAIR", "video_segments_video6_small_3")   # must be ignored
    for d in (masks, images, flair, hil, other):
        os.makedirs(d)
    shark, false = rect(40, 80, 50, 110), rect(120, 160, 180, 240)
    for k, gt_frame in enumerate((4930, 4940, 4955)):                       # clip frames 10, 20, 35 (start 4920)
        gt = np.zeros((H, W, 3), dtype=np.uint8)
        gt[shark] = ev.GT_COLORS[0]
        Image.fromarray(gt).save(os.path.join(masks, f"video5_{gt_frame:05d}.png"))
        img = np.full((H, W, 3), 30, dtype=np.uint8)
        img[shark] = 255                                                     # shark-like patch
        img[false] = 60                                                      # dim false patch
        Image.fromarray(img).save(os.path.join(images, f"video5_{gt_frame:05d}.jpg"), quality=95)
        clip_frame = gt_frame - 4920
        np.save(os.path.join(flair, f"video5_frame_{clip_frame}.npy"), {1: shark[None], 2: false[None]})
        np.save(os.path.join(hil, f"frame_{clip_frame}.npy"), {1: shark[None]})
        np.save(os.path.join(other, f"video6_frame_{clip_frame}.npy"), {9: shark[None]})


def test_paths_and_discovery():
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        assert gr.flair_dir(root, "video5").endswith("video_segments_video5_small_3")
        assert gr.hil_dir(root, "video5").endswith("video_segments_HiL_video5_small")
        assert os.path.isdir(gr.gt_dir(root, "video5")) and os.path.isdir(gr.img_dir(root, "video5"))
        try:
            gr.flair_dir(root, "video3")
            raise AssertionError("missing video must raise")
        except FileNotFoundError:
            pass


def test_self_test_against_summary():
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        try:
            gr.self_test(root, os.path.join(root, "missing.csv"), ["video5"])
            raise AssertionError("a missing summary csv must fail loudly, not pass silently")
        except FileNotFoundError:
            pass
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        flair = ev.evaluate_records(ev.build_records("video5", gr.gt_dir(root, "video5"), gr.flair_dir(root, "video5")))
        hil = ev.evaluate_records(ev.build_records("video5", gr.gt_dir(root, "video5"), gr.hil_dir(root, "video5")))
        path = os.path.join(root, "summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "video", "frames", "dice_authors", "dice_union"])
            w.writerow(["FLAIR_3", "video5", 3, round(flair["authors_dice"], 4), round(flair["union_dice"], 4)])
            w.writerow(["HiL", "video5", 3, round(hil["authors_dice"], 4), round(hil["union_dice"], 4)])
        computed, problems = gr.self_test(root, path, ["video5"])
        assert problems == [], problems
        assert abs(computed[("FLAIR_3", "video5")]["authors_dice"] - 1.0) < 1e-9
        assert computed[("FLAIR_3", "video5")]["union_dice"] < computed[("HiL", "video5")]["union_dice"]


def test_run_video_gate_passes_and_saves():
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        out = os.path.join(root, "out")
        res = gr.run_video("video5", root, fake_scorer, out_dir=out)
        gr.print_summary(res)
        base, med = res["baseline"], res["median_clip"]
        assert base["extra_tracks_retained"] == 1 and med["extra_tracks_retained"] == 0
        assert res["median_clip_kept"] == [1] and res["q25_clip_kept"] == [1]
        assert res["median_clip_gate"]["passed"]
        assert med["union_dice"] > base["union_dice"] and med["authors_dice"] == base["authors_dice"]
        assert abs(res["oracle"]["union_dice"] - med["union_dice"]) < 1e-9
        rows = {r["track_id"]: r for r in res["tracks"]}
        assert rows[1]["shark"] and not rows[2]["shark"] and rows[1]["n_obs"] == 3
        assert os.path.exists(os.path.join(out, "video5_tracks.csv"))
        assert os.path.exists(os.path.join(out, "video5_summary.json"))


def test_crop_montage_writes_png():
    import matplotlib
    matplotlib.use("Agg")
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        out = os.path.join(root, "diag", "m.png")
        fig = gr.crop_montage("video5", root, [1, 2], fake_scorer, n=4, out_path=out)
        assert os.path.getsize(out) > 1000
        assert len(fig.axes) == 8                       # 2 tracks x 4 columns (extra columns are blank)
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_spearman_and_ranks():
    approx = lambda a, b: abs(a - b) < 1e-9
    assert approx(gr.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert approx(gr.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
    assert np.isnan(gr.spearman([1, 1, 1], [1, 2, 3]))                      # constant input
    assert np.isnan(gr.spearman([1, 2], [1, 2]))                            # too few points
    assert list(gr._rank([5, 5, 1])) == [2.5, 2.5, 1.0]                     # ties share the mean rank


def test_auc():
    assert gr.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert gr.auc([0.1], [0.9]) == 0.0
    assert gr.auc([0.5], [0.5]) == 0.5
    assert np.isnan(gr.auc([], [0.5]))
    assert abs(gr.auc([1, 3], [2]) - 0.5) < 1e-12


def test_print_probe_reports_auc(capsys=None):
    import io, contextlib
    rows = []
    for tid, shark, p_raw, p_pad in [(1, True, 0.6, 0.99), (2, False, 0.9, 0.95), (3, False, 0.8, 0.9)]:
        for f in range(3):
            rows.append({"video": "v", "track_id": tid, "shark": shark, "frame": f, "h": 10, "w": 5,
                         "aspect": 2.0, "p_raw": p_raw + 0.01 * f, "p_pad": p_pad})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gr.print_probe(rows)
    out = buf.getvalue()
    assert "raw: crops 0.00, track medians 0.00" in out      # shark scored lower than extras: inverted
    assert "pad: crops 1.00, track medians 1.00" in out      # shark scored higher: perfect


def test_probe_crops_records_both_scorers(capsys=None):
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        rows = gr.probe_crops("video5", root, {"raw": fake_scorer, "pad": lambda cs: [0.5] * len(cs)})
        assert len(rows) == 6                                               # 2 tracks x 3 frames
        assert {r["track_id"] for r in rows} == {1, 2}
        assert all(r["aspect"] >= 1.0 and "p_raw" in r and "p_pad" in r for r in rows)
        assert all(r["p_pad"] == 0.5 for r in rows)
        assert next(r for r in rows if r["track_id"] == 1)["shark"] is True
        gr.print_probe(rows)


def test_size_mismatch_is_caught():
    with tempfile.TemporaryDirectory() as root:
        build_tree(root)
        imgs = gr.img_dir(root, "video5")
        for n in os.listdir(imgs):
            Image.fromarray(np.zeros((H // 2, W // 2, 3), np.uint8)).save(os.path.join(imgs, n))
        try:
            gr.run_video("video5", root, fake_scorer)
            raise AssertionError("image/mask size mismatch must raise")
        except ValueError as e:
            assert "mask shape" in str(e)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
