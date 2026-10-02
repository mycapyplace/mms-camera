"""Train an RF-DETR Nano baseline on the frozen FSOCO train/validation split."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

from converters.convert_fsoco import CLASSES, EXPECTED_BOXES, EXPECTED_IMAGES, EXPECTED_MANIFEST_SHA256
from converters.convert_fsoco_coco import CATEGORIES, FOLDERS


RFDETR_VERSION = "1.11.1"


def validate_dataset(dataset: Path) -> None:
    """Audit train/valid before training; do not load the held-out test split."""
    if (dataset / "source_manifest.sha256").read_text().split()[0] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Dataset was not made from the frozen manifest")
    for split in ("train", "val"):
        folder = dataset / FOLDERS[split]
        data = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        if data["info"]["manifest_sha256"] != EXPECTED_MANIFEST_SHA256 or data["categories"] != CATEGORIES:
            raise ValueError(f"{split}: manifest or class mapping mismatch")
        images = {image["id"]: image for image in data["images"]}
        if len(images) != len(data["images"]) or len(images) != EXPECTED_IMAGES[split]:
            raise ValueError(f"{split}: wrong image count or duplicate IDs")
        filenames = [image["file_name"] for image in images.values()]
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"{split}: duplicate filenames")
        for filename in filenames:
            if Path(filename).name != filename or not (folder / filename).is_file():
                raise ValueError(f"{split}: invalid or missing image {filename}")
        counts = Counter(annotation["category_id"] for annotation in data["annotations"])
        if set(counts) - set(range(len(CLASSES))) or tuple(counts[i] for i in range(len(CLASSES))) != EXPECTED_BOXES[split]:
            raise ValueError(f"{split}: box count or class ID mismatch")
        if len({ann["id"] for ann in data["annotations"]}) != len(data["annotations"]):
            raise ValueError(f"{split}: duplicate annotation IDs")
        for annotation in data["annotations"]:
            image = images[annotation["image_id"]]
            x, y, width, height = annotation["bbox"]
            if not (0 <= x < x + width <= image["width"] and 0 <= y < y + height <= image["height"]
                    and annotation["area"] == width * height and annotation["iscrowd"] == 0):
                raise ValueError(f"{split}: invalid COCO box")
        print(f"Verified {split}: {len(images)} images, {sum(counts.values())} boxes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--smoke", action="store_true", help="One full epoch; verifies training and validation")
    parser.add_argument("--check-only", action="store_true", help="Audit train/valid without importing RF-DETR")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.grad_accum_steps, args.epochs) < 1 or args.workers < 0:
        parser.error("Batch size, accumulation and epochs must be positive; workers must be nonnegative")
    validate_dataset(args.dataset)
    if args.check_only:
        return
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Run directory already exists: {args.output}")
    if importlib.metadata.version("rfdetr") != RFDETR_VERSION:
        raise RuntimeError(f"This script is verified against rfdetr=={RFDETR_VERSION}")
    import torch
    from rfdetr import RFDETRNano

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment; check the GPU installation")
    config = dict(
        dataset_dir=str(args.dataset.resolve()), dataset_file="roboflow",
        output_dir=str(args.output.resolve()), resolution=640, device="cuda:0",
        epochs=1 if args.smoke else args.epochs, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, num_workers=args.workers,
        seed=42, lr=1e-4, lr_encoder=1.5e-4,
        early_stopping=not args.smoke, early_stopping_patience=20,
        early_stopping_min_delta=0.0, early_stopping_use_ema=True,
        use_ema=True, best_model_metric="map", run_test=False,
        multi_scale=False, expanded_scales=False, scale_jitter=False,
        tensorboard=False, wandb=False, mlflow=False,
        log_per_class_metrics=True, checkpoint_interval=100,
    )
    # Validate API fields before downloading model weights or reserving a run directory.
    from rfdetr.config import TrainConfig
    TrainConfig(**{k: v for k, v in config.items() if k not in ("resolution", "device")})
    args.output.mkdir(parents=True)
    provenance = {
        "variant": "RFDETRNano", "smoke": args.smoke, "train_arguments": config,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256, "class_names": list(CLASSES),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": subprocess.check_output(
            [os.sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    print(f"Effective batch: {args.batch_size * args.grad_accum_steps}; held-out test disabled", flush=True)
    model = RFDETRNano(resolution=640, device="cuda")
    model.train(**config)


if __name__ == "__main__":
    main()
