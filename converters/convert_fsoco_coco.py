"""Convert the frozen FSOCO manifest to RF-DETR's Roboflow COCO layout."""

import argparse
import json
import os
from pathlib import Path

from converters.convert_fsoco import CLASSES, SPLITS, load_manifest


FOLDERS = {"train": "train", "val": "valid", "test": "test"}
CATEGORIES = [{"id": i, "name": name, "supercategory": "cone"}
              for i, name in enumerate(CLASSES)]


def convert(manifest: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    digest, entries, images, boxes = load_manifest(manifest)
    datasets = {split: {"info": {"description": "Frozen FSOCO split v1", "manifest_sha256": digest},
                        "licenses": [], "images": [], "annotations": [], "categories": CATEGORIES}
                for split in SPLITS}
    links = []
    for image_id, (split, image, stem, width, height, rectangles) in enumerate(entries, 1):
        dataset = datasets[split]
        filename = f"{stem}{image.suffix}"
        dataset["images"].append({"id": image_id, "file_name": filename,
                                  "width": width, "height": height})
        links.append((image.resolve(), output / FOLDERS[split] / filename))
        for class_id, x1, y1, x2, y2 in rectangles:
            box_width, box_height = x2 - x1, y2 - y1
            dataset["annotations"].append({
                "id": len(dataset["annotations"]) + 1, "image_id": image_id,
                "category_id": class_id, "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height, "iscrowd": 0,
            })

    output.mkdir(parents=True)
    for split in SPLITS:
        folder = output / FOLDERS[split]
        folder.mkdir()
        (folder / "_annotations.coco.json").write_text(
            json.dumps(datasets[split], separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
    for source, link in links:
        os.symlink(source, link)
    if any(not link.is_file() for _, link in links):
        raise ValueError("Conversion left a broken image symlink")
    (output / "source_manifest.sha256").write_text(
        f"{digest}  {manifest.resolve()}\n", encoding="utf-8"
    )
    for split in SPLITS:
        print(f"{FOLDERS[split]}: {images[split]} images, {sum(boxes[split].values())} boxes")
    print(f"Dataset: {output.resolve()}\nCategory IDs 0–4: {', '.join(CLASSES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, non-existent output directory")
    args = parser.parse_args()
    convert(args.manifest, args.output)
