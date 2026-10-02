"""
Used to convert FSOCO Supervisely into YOLO-compatible format.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


CLASSES = (
    "yellow_cone",
    "blue_cone",
    "orange_cone",
    "large_orange_cone",
    "unknown_cone",
)
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
SPLITS = ("train", "val", "test")
VAL_TEAMS = {"frt", "ulm"}
TEST_TEAMS = {"mms", "dtu", "bme"}
EXPECTED_MANIFEST_SHA256 = "5af806b2fa469e5b5b544d195dcd18c21c56a65a3f0c5179bf0bbf14c8cba09d"
EXPECTED_IMAGES = {"train": 8997, "val": 1137, "test": 1115}
EXPECTED_BOXES = {
    "train": (79763, 72939, 22041, 5160, 5334),
    "val": (4991, 4735, 1591, 1180, 1057),
    "test": (6743, 6936, 1077, 1628, 890),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def convert(manifest: Path, output: Path) -> None:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA-256 mismatch: {digest}; expected {EXPECTED_MANIFEST_SHA256}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    if "\n" in str(output) or "'" in str(output):
        raise ValueError("Output path cannot contain a quote or newline")

    # Validate the complete input before creating any output.
    entries = []
    seen = set()
    images = Counter()
    boxes = {split: Counter() for split in SPLITS}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        split, team = row["split"], row["team"]
        if split not in SPLITS or team == "prom":
            raise ValueError(f"Row {line_number}: invalid split or excluded team")
        expected_split = "val" if team in VAL_TEAMS else "test" if team in TEST_TEAMS else "train"
        if split != expected_split:
            raise ValueError(f"Row {line_number}: {team} must be in {expected_split}")
        image, annotation = Path(row["image"]), Path(row["annotation"])
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Row {line_number}: missing {image} or {annotation}")
        if image.suffix.lower() not in IMAGE_EXTENSIONS or annotation.name != image.name + ".json":
            raise ValueError(f"Row {line_number}: unexpected image/annotation filename")
        if image.parent.name != "img" or annotation.parent.name != "ann":
            raise ValueError(f"Row {line_number}: unexpected raw folder layout")
        if image.parent.parent != annotation.parent.parent or image.parent.parent.name != team:
            raise ValueError(f"Row {line_number}: team/path mismatch")
        stem = f"{team}__{image.stem}"
        key = (split, stem.lower())
        if key in seen:
            raise ValueError(f"Row {line_number}: duplicate output label stem {key}")
        seen.add(key)

        data = json.loads(annotation.read_text(encoding="utf-8"))
        width, height = data["size"]["width"], data["size"]["height"]
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError(f"Row {line_number}: invalid image dimensions")
        labels = []
        for obj in data.get("objects", []):
            name = obj.get("classTitle")
            if obj.get("geometryType") != "rectangle" or name not in CLASS_IDS:
                raise ValueError(f"Row {line_number}: unexpected class/geometry {name}")
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            if not (all(math.isfinite(v) for v in (x1, y1, x2, y2))
                    and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Row {line_number}: invalid rectangle {name}")
            labels.append(f"{CLASS_IDS[name]} {(x1 + x2) / (2 * width):.8f} "
                          f"{(y1 + y2) / (2 * height):.8f} "
                          f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}")
            boxes[split][name] += 1
        images[split] += 1
        entries.append((split, image, stem, labels))

    if dict(images) != EXPECTED_IMAGES:
        raise ValueError(f"Unexpected image counts: {dict(images)}")
    for split in SPLITS:
        actual = tuple(boxes[split][name] for name in CLASSES)
        if actual != EXPECTED_BOXES[split]:
            raise ValueError(f"Unexpected {split} box counts: {actual}")

    output.mkdir(parents=True)  # refuses an existing output; never clears data
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for split, image, stem, labels in entries:
        link = output / "images" / split / f"{stem}{image.suffix}"
        os.symlink(image.resolve(), link)
        (output / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )

    # YAML syntax for an absolute Unix path; all source images remain in place.
    yaml = f"path: '{output.resolve()}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "source_manifest.sha256").write_text(
        f"{digest}  {manifest.resolve()}\n", encoding="utf-8"
    )
    for split in SPLITS:
        print(f"{split}: {images[split]} images, {sum(boxes[split].values())} boxes")
    print(f"Dataset: {output.resolve()}\nYOLO config: {output.resolve() / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, non-existent output directory")
    args = parser.parse_args()
    convert(args.manifest, args.output)
