# mms-camera
Camera-Based Vision System for a Formula Student Driverless Vehicle by Aden T

## Current project

1. Fine-tune YOLO26n and RF-DETR on the same frozen FSOCO train split,
   select checkpoints on FSOCO validation, and compare them on a separately
   human-reviewed Monash Motorsport evaluation set.
2. Assess whether Grounding DINO helps annotate Monash images faster while
   maintaining human-reviewed label quality. It is an annotation assistant,
   not one of the two supervised detectors being compared.

EfficientDet is outside the current scope. The `mms` team inside FSOCO is
**Munich Motorsport**, not Monash Motorsport. The previous preliminary report
and the Week 5 three-model study guide in `docs/` are retained as historical
research, not the current experimental plan. The current results and resume
guide are in the FIT4702 Notion teamspace:

- [Results and data](https://app.notion.com/p/3ede6e5bcd038153b903cc1fbe2cb344)
- [FYP progress and resume guide](https://app.notion.com/p/3eae6e5bcd0381e48f01dabf4a69f704)

## FSOCO conversion and experiment records

The YOLO26n baseline was trained on the frozen split and independently
validated on FSOCO val at **mAP50-95 = 0.416** (not a Monash result). The
converter in `scripts/convert_fsoco.py` uses symbolic links to the original
images instead of copying them. Keep the raw images in place. Its synthetic
regression test runs with `python -m unittest discover -s tests -v`.

For reproducibility, keep small files copied from the server in these paths:

```text
manifests/fsoco_split_v1.jsonl
manifests/fsoco_split_v1.sha256
experiments/yolo26n-fsoco-baseline-v1/args.yaml
experiments/yolo26n-fsoco-baseline-v1/results.csv
```

The manifest records absolute **server** paths; it is a frozen record, not a
portable dataset. Its SHA-256 is
`5af806b2fa469e5b5b544d195dcd18c21c56a65a3f0c5179bf0bbf14c8cba09d`.
Keep checkpoints locally in ignored `artifacts/` (and back them up elsewhere),
not in Git history. `best.pt` is needed for inference and Monash evaluation;
`last.pt` is useful if training must resume.

The following command documents the already-completed conversion. The script
was transferred to the server's repo root as `convert_fsoco.py`; the version
in this repository is `scripts/convert_fsoco.py`. The converter refuses to
overwrite the existing output, so use a **new** output path if rebuilding:

```bash
cd /home/atrantan/mms-camera
df -h "$HOME"
python3 /home/atrantan/mms-camera/convert_fsoco.py \
  --manifest /home/atrantan/fyp-data/manifests/fsoco_split_v1.jsonl \
  --output /home/atrantan/fyp-data/yolo-fsoco-rebuild
```

The script checks the frozen manifest SHA-256, whole-team assignments, image
and annotation paths, rectangle geometry, and the expected class and split
counts before writing. It refuses to overwrite an output directory. If it is
interrupted after creating the output, investigate that partial output before
choosing a new output location; never point it at an existing dataset.
`data.yaml` points to `images/{train,val,test}` and labels are under
`labels/{train,val,test}`. Empty-annotation images receive empty `.txt` files.
The class IDs are yellow=0, blue=1, orange=2, large orange=3, unknown=4.

Verify the result on the server before training:

```bash
find /home/atrantan/fyp-data/yolo-fsoco-v1/images -type l | wc -l  # 11249
find /home/atrantan/fyp-data/yolo-fsoco-v1/labels -type f | wc -l  # 11249
find /home/atrantan/fyp-data/yolo-fsoco-v1/images -xtype l  # no broken links
cat /home/atrantan/fyp-data/yolo-fsoco-v1/data.yaml
df -h "$HOME"
```

Use train and validation during model development. Keep the FSOCO test split
and the final human-reviewed Monash evaluation set out of model selection.
