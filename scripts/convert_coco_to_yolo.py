import argparse
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] in pixels
    to normalized YOLO [x_center, y_center, width, height].
    """
    x, y, w, h = bbox

    x_center = x + w / 2
    y_center = y + h / 2

    return (
        x_center / image_width,
        y_center / image_height,
        w / image_width,
        h / image_height,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO/Objects365 annotations to YOLO format."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to COCO/Objects365 JSON annotation file",
    )
    parser.add_argument(
        "--output-labels-dir",
        required=True,
        help="Directory where YOLO .txt labels will be written",
    )
    parser.add_argument(
        "--skip-crowd",
        action="store_true",
        help="Skip annotations marked as crowd",
    )
    parser.add_argument(
        "--category-mode",
        choices=["contiguous", "original"],
        default="contiguous",
        help=(
            "Use contiguous class IDs starting from 0, or preserve original COCO category IDs. "
            "For YOLO training, contiguous is usually preferred."
        ),
    )

    args = parser.parse_args()

    coco = COCO(args.annotations)
    output_labels_dir = Path(args.output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    cat_ids = sorted(coco.getCatIds())

    if args.category_mode == "contiguous":
        cat_id_to_class_id = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    else:
        cat_id_to_class_id = {cat_id: cat_id for cat_id in cat_ids}

    image_ids = coco.getImgIds()

    for image_id in tqdm(image_ids, desc="Converting COCO annotations to YOLO"):
        image_info = coco.loadImgs(image_id)[0]

        image_width = image_info["width"]
        image_height = image_info["height"]
        file_name = image_info["file_name"]

        label_path = output_labels_dir / Path(file_name).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)

        lines = []

        for ann in anns:
            if args.skip_crowd and ann.get("iscrowd", 0):
                continue

            bbox = ann.get("bbox")
            if bbox is None:
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            category_id = ann["category_id"]
            class_id = cat_id_to_class_id[category_id]

            x_center, y_center, box_width, box_height = coco_bbox_to_yolo(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )

            lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

        with label_path.open("w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    print(f"Done. YOLO labels saved to: {output_labels_dir}")


if __name__ == "__main__":
    main()
