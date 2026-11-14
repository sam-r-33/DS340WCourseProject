"""Self-contained tests for evaluation.py and track_validation.py (synthetic data, no GPU, no CLIP download).

Run:  python3 test_modules.py
"""
import csv
import os
import tempfile

import numpy as np
from PIL import Image

import evaluation as ev
import track_validation as tv

H = W = 200


def rect(r0, r1, c0, c1, shape=(H, W)):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


# ----------------------------------------------------------------------------- evaluation
def test_dice_and_guard():
    a = rect(0, 10, 0, 10)
    approx(ev.dice(a, a), 1.0)
    approx(ev.dice(a, rect(0, 10, 5, 15)), 0.5)
    approx(ev.dice(np.zeros((5, 5), bool), np.zeros((5, 5), bool)), 0.0)  # zero-division guard


def test_decode_gt():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[10:30, 10:30] = ev.GT_COLORS[0]
    img[100:120, 100:120] = ev.GT_COLORS[1]
    masks = ev.decode_gt(img)
    assert len(masks) == 2 and masks[0].sum() == 400 and masks[1].sum() == 400
    only_one = np.zeros((H, W, 3), dtype=np.uint8)
    only_one[0:5, 0:5] = ev.GT_COLORS[1]
    assert len(ev.decode_gt(only_one)) == 1


def test_schema_normalisation():
    arr = rect(10, 30, 10, 30)[None]                      # released style: (1, H, W) boolean
    poly = [[[10, 10], [29, 10], [29, 29], [10, 29]]]     # our notebooks: list of polygons
    out = ev.to_mask_dict({1: arr, 2: poly}, image_shape=(H, W))
    assert out[1].shape == (H, W) and out[2].shape == (H, W)
    assert out[1].sum() == 400
    assert ev.dice(out[1], out[2]) > 0.85                 # polygon fill is ~ the same square
    try:
        ev.to_mask_dict({2: poly})
        raise AssertionError("polygons without image_shape must fail")
    except ValueError:
        pass


def _records():
    gt = [rect(10, 30, 10, 30)]
    a = ev.FrameRecord("v", 1, gt, {1: rect(10, 30, 10, 30), 2: rect(100, 120, 100, 120)})
    b = ev.FrameRecord("v", 2, gt, {1: rect(10, 30, 20, 40), 2: rect(100, 120, 100, 120)})
    c = ev.FrameRecord("v", 3, gt, None)  # missing prediction file
    return [a, b, c]


def test_metrics_unpruned():
    recs = _records()
    m = ev.evaluate_records(recs)
    approx(m["authors_dice"], (1.0 + 0.5 + 0.0) / 3)
    approx(m["union_dice"], (2 / 3 + 1 / 3 + 0.0) / 3)
    approx(m["recall"], 2 / 3)                            # Dice 0.5 counts as found (>=), missing frame does not
    assert m["missing_frames"] == 1 and m["frames_scored"] == 3
    assert (m["tracks_present"], m["tracks_retained"]) == (2, 2)
    assert (m["shark_tracks_retained"], m["extra_tracks_retained"]) == (1, 1)


def test_labels_oracle_and_pruning():
    recs = _records()
    labels = ev.label_tracks(recs)
    assert labels[1]["shark"] and not labels[2]["shark"]
    assert ev.oracle_keep(labels) == {1}
    m = ev.evaluate_records(recs, keep={1})
    approx(m["authors_dice"], 0.5)                        # best-match Dice cannot improve by removing masks
    approx(m["union_dice"], (1.0 + 0.5 + 0.0) / 3)        # union Dice does
    assert m["extra_tracks_retained"] == 0 and m["shark_tracks_retained"] == 1


def test_gate():
    recs = _records()
    base = ev.evaluate_records(recs)
    good = ev.evaluate_records(recs, keep={1})
    g = ev.gate_check(base, good)
    assert g["passed"] and g["extra_removed_frac"] == 1.0
    assert not ev.gate_check(base, base)["passed"]        # removed nothing
    lost_shark = ev.evaluate_records(recs, keep={2})      # drops the real shark -> recall + Dice collapse
    g2 = ev.gate_check(base, lost_shark)
    assert not g2["ok_recall"] and not g2["ok_dice"] and not g2["passed"]


def test_files_roundtrip_and_offsets():
    with tempfile.TemporaryDirectory() as d:
        gt_dir, pred_dir = os.path.join(d, "gt"), os.path.join(d, "pred")
        os.makedirs(gt_dir), os.makedirs(pred_dir)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[10:30, 10:30] = ev.GT_COLORS[0]
        Image.fromarray(img).save(os.path.join(gt_dir, "video1_2410.png"))   # clip frame 10 (start 2400)
        Image.fromarray(img).save(os.path.join(gt_dir, "video1_2420.png"))   # clip frame 20: prediction missing
        Image.fromarray(img).save(os.path.join(gt_dir, "video2_5170.png"))   # other video: must be ignored
        poly = {7: [[[10, 10], [29, 10], [29, 29], [10, 29]]]}
        np.save(os.path.join(pred_dir, "video1_frame_10.npy"), poly)         # polygon schema, prefixed name
        recs = ev.build_records("video1", gt_dir, pred_dir, image_shape=(H, W))
        assert [r.gt_frame for r in recs] == [2410, 2420]
        assert recs[0].pred is not None and recs[1].pred is None
        m = ev.evaluate_records(recs)
        assert m["missing_frames"] == 1
        assert m["authors_dice"] > 0.4                                       # (~0.95 + 0) / 2
        # unprefixed filename + boolean schema + full-video numbering
        np.save(os.path.join(pred_dir, "frame_2420.npy"), {3: rect(10, 30, 10, 30)[None]})
        recs2 = ev.build_records("video1", gt_dir, pred_dir, image_shape=(H, W), full_video_numbering=True)
        assert recs2[1].pred is not None and recs2[0].pred is None
    assert ev.parse_gt_name("video1_2400.png") == ("video1", 2400)
    assert ev.parse_gt_name("blacktip_17.png") == ("blacktip", 17)


def test_check_against_summary():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "video", "frames", "dice_authors", "dice_union"])
            w.writerow(["FLAIR_3", "video2", 38, 0.795, 0.1239])
        ok = {("FLAIR_3", "video2"): {"authors_dice": 0.7951, "union_dice": 0.1238}}
        bad = {("FLAIR_3", "video2"): {"authors_dice": 0.70, "union_dice": 0.1238}}
        assert ev.check_against_summary(ok, path) == []
        assert len(ev.check_against_summary(bad, path)) == 1


def test_packed_mask_matches_plain():
    a, b = rect(10, 30, 10, 30), rect(10, 30, 20, 40)
    pa = ev.PackedMask(a[None])                                             # (1, H, W) input, like released masks
    assert pa.shape == (H, W) and pa.sum() == 400 and pa.any()
    assert np.array_equal(np.asarray(pa), a)
    approx(ev.dice(pa, b), ev.dice(a, b))
    recs = _records()
    packed = [ev.FrameRecord(r.video, r.gt_frame, r.gt,
                             None if r.pred is None else {k: ev.PackedMask(v) for k, v in r.pred.items()})
              for r in recs]
    plain_m, packed_m = ev.evaluate_records(recs), ev.evaluate_records(packed)
    for k in plain_m:
        if isinstance(plain_m[k], float):
            approx(plain_m[k], packed_m[k])
        else:
            assert plain_m[k] == packed_m[k], k
    # mask spans 10..29; padded by 25 -> 0..54 inclusive (lower side clipped at 0) = 55 px
    assert tv.crop_for_mask(np.zeros((H, W, 3), np.uint8), pa).shape[:2] == (55, 55)


def test_names_without_prefix_and_zero_padded():
    assert ev.parse_gt_name("0047.png", default_video="blacktip") == ("blacktip", 47)
    try:
        ev.parse_gt_name("0047.png")
        raise AssertionError("needs default_video")
    except ValueError:
        pass
    assert "02589.npy" in ev.pred_candidates("video1", 2589)
    assert "frame_0047.jpg" in ev.image_candidates("blacktip", 47)
    assert "video1_02589.jpg" in ev.image_candidates("video1", 2589)


# ----------------------------------------------------------------------------- track validation
def test_pad_to_square():
    crop = np.full((10, 4, 3), 100, dtype=np.uint8)
    crop[:, 0] = 200
    out = tv.pad_to_square(crop)
    assert out.shape == (10, 10, 3) and out.dtype == np.uint8
    assert np.array_equal(out[:, 3:7], crop)                               # content untouched, centred
    mean = crop.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    assert np.array_equal(out[0, 0], mean) and np.array_equal(out[5, 9], mean)
    sq = np.zeros((6, 6, 3), np.uint8)
    assert tv.pad_to_square(sq) is sq                                      # already square -> unchanged
    wide = np.zeros((4, 9, 3), np.uint8)
    assert tv.pad_to_square(wide).shape == (9, 9, 3)


def test_crop_padding_and_clipping():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    corner = rect(0, 10, 0, 10, (100, 100))
    assert tv.crop_for_mask(img, corner).shape[:2] == (35, 35)            # box (0,0,9,9) -> 0..34 inclusive
    centre = rect(50, 60, 50, 60, (100, 100))
    assert tv.crop_for_mask(img, centre).shape[:2] == (60, 60)            # 25..84 inclusive
    assert tv.crop_for_mask(img, np.zeros((100, 100), bool)) is None


def test_sampling():
    s = tv.sample_frames(list(range(25)))
    assert len(s) == 10 and s[0] == 0 and s[-1] == 24 and s == sorted(set(s))
    assert tv.sample_frames([3, 1, 2]) == [1, 2, 3]


def test_aggregates():
    probs = [0.99] * 7 + [0.2] * 3
    approx(tv.aggregate(probs, "median"), 0.99)
    approx(tv.aggregate(probs, "q25"), 0.2 + 0.25 * 0.79, 1e-9)           # linear interpolation, index 2.25
    assert np.isnan(tv.aggregate([], "median"))


def _toy(n_frames=12):
    images = {f: np.zeros((100, 100, 3), dtype=np.uint8) for f in range(n_frames)}
    masks = {}
    for f in range(n_frames):
        images[f][10:20, 10:20] = 255      # track 1: bright patch ("shark-like")
        images[f][60:70, 60:70] = 50       # track 2: dim patch ("not a shark")
        masks[(1, f)] = rect(10, 20, 10, 20, (100, 100))
        masks[(2, f)] = rect(60, 70, 60, 70, (100, 100))
        masks[(3, f)] = np.zeros((100, 100), bool)                        # empty everywhere
    return images, masks


def _fake_scorer(crops):
    return [0.99 if c.max() > 200 else 0.3 for c in crops]


def test_validator_median_keeps_shark_removes_extra():
    images, masks = _toy()
    tf = {1: list(range(12)), 2: list(range(12)), 3: list(range(12))}
    kept, rows = tv.validate_tracks(tf, lambda t, f: masks[(t, f)], lambda f: images[f], _fake_scorer,
                                    variant="median_clip")
    assert kept == [1]
    by = {r["track_id"]: r for r in rows}
    assert by[1]["n_obs"] == 10 and by[2]["n_obs"] == 10                  # capped at 10 of 12 frames
    assert by[3]["n_obs"] == 0 and not by[3]["keep_median_clip"]          # nothing to score -> dropped
    approx(by[1]["median"], 0.99)
    approx(by[2]["frac_ge_095"], 0.0)


def test_short_track_uses_all_observations():
    images, masks = _toy()
    kept, rows = tv.validate_tracks({1: [0, 1, 2]}, lambda t, f: masks[(t, f)], lambda f: images[f], _fake_scorer)
    assert kept == [1] and rows[0]["n_obs"] == 3


def test_q25_is_stricter_than_median():
    images, masks = _toy()
    seq = iter([0.99] * 7 + [0.2] * 3)

    def scorer(crops):
        return [next(seq) for _ in crops]

    tf = {1: list(range(10))}
    get_mask, get_img = (lambda t, f: masks[(t, f)]), (lambda f: images[f])
    rows = tv.score_tracks(tf, get_mask, get_img, scorer)
    r = rows[0]
    assert r["keep_median_clip"] and not r["keep_q25_clip"]
    approx(r["frac_ge_095"], 0.7)


def test_switch():
    images, masks = _toy()
    tf = {1: list(range(12)), 2: list(range(12))}
    kept, rows = tv.select_tracks("manual", [2, 1])
    assert kept == [1, 2] and rows is None
    kept, rows = tv.select_tracks("median_clip", [], track_frames=tf, get_mask=lambda t, f: masks[(t, f)],
                                  get_image=lambda f: images[f], scorer=_fake_scorer)
    assert kept == [1] and rows is not None
    try:
        tv.select_tracks("bogus", [], track_frames=tf, get_mask=None, get_image=None, scorer=None)
        raise AssertionError("unknown variant must fail")
    except ValueError:
        pass


def test_end_to_end_validator_into_scorer():
    """Validator feeds the scorer: keep only what CLIP keeps, then check the pre-specified gate."""
    images, masks = _toy(12)
    gt = [rect(10, 20, 10, 20, (100, 100))]
    records = [ev.FrameRecord("toy", f, gt, {1: masks[(1, f)], 2: masks[(2, f)]}) for f in range(12)]
    tf = {t: list(range(12)) for t in (1, 2)}
    kept, _ = tv.validate_tracks(tf, lambda t, f: masks[(t, f)], lambda f: images[f], _fake_scorer)
    base = ev.evaluate_records(records)
    after = ev.evaluate_records(records, keep=kept)
    assert ev.gate_check(base, after)["passed"]
    assert after["union_dice"] > base["union_dice"] and after["authors_dice"] == base["authors_dice"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nall {len(tests)} tests passed")
