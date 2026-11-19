# demo_data: inputs for the technical demo (notebook 04, blacktip clip)

This folder holds the small amount of third-party data needed to run `notebooks/04_FLAIR_track_validation.ipynb` on the example blacktip reef shark clip without downloading the full ~45 GB dataset. **None of it is our own work.** It is included unchanged, with credit to its authors. See "Technical demo" in the main `README.md` for how it is used.

## Contents

| Path | What it is | Source and license |
|---|---|---|
| `example_video_blacktip.mp4` | The example drone video used in the FLAIR repository (4,889,487 bytes). The demo uses its first 8 seconds (237 frames). | Conservation Technology Group, FLAIR repository, https://github.com/conservation-technology-group/FLAIR (MIT). Also in `FLAIR_code.zip` in the Zenodo code archive, DOI 10.5281/zenodo.16056059 (MIT). |
| `trimmed_videos/sampled_frames/masks/blacktip_masks/*.png` | Ground-truth segmentation masks for the blacktip clip: 25 annotated frames, named by frame number (0047.png to 0544.png). They are used only when `EVAL_VIDEO = "blacktip"` is set in the notebook's Configuration cell. | The dataset released with the FLAIR paper, Zenodo record DOI 10.5281/zenodo.15845961 (CC BY 4.0). The folder path mirrors `UnzippedData/trimmed_videos/sampled_frames/masks/blacktip_masks/` in that dataset. |

The 25 masks were copied as they are, not selected by us. The 8-second demo clip covers frames 0 to 236, so 12 of the 25 masks (frames 0047 to 0217) fall inside it and are scored; the other 13 are not used in the demo. This agrees with `results/track_validation/e2e_blacktip/validation/evaluation.json`, which reports 12 annotated frames.

## Credit

Lalgudi et al. (2025), *Zero-shot shark tracking and biometrics from aerial imagery*, Methods in Ecology and Evolution (arXiv:2501.05717), and the dataset and code released with it. If you reuse these files, cite the paper and the Zenodo record, and follow the license listed for each file above.
