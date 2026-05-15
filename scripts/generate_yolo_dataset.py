import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from widefov.annotations.bbox import is_valid_yolo_box
from widefov.transforms.registry import available_transforms, get_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_to_coco_xywh(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO xywh to pixel COCO xywh."""
    width_px = box_width * image_width
    height_px = box_height * image_height
    x_px = (x_center * image_width) - width_px / 2
    y_px = (y_center * image_height) - height_px / 2

    return x_px, y_px, width_px, height_px


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []

    if not label_path.exists():
        return labels

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            labels.append((class_id, x_center, y_center, width, height))

    return labels


def write_yolo_labels(
    label_path: Path,
    labels: list[tuple[int, float, float, float, float]],
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w") as f:
        for class_id, x_center, y_center, width, height in labels:
            f.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


def find_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a transformed wide-FoV/fisheye YOLO-format dataset."
    )
    parser.add_argument("--images-dir", required=True, help="Input image directory")
    parser.add_argument("--labels-dir", required=True, help="Input YOLO label directory")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument(
        "--transform",
        required=True,
        choices=available_transforms(),
        help="Transform to apply",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip images without label files or without valid transformed labels",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform(args.transform)
    suffix = transform.suffix if hasattr(transform, "suffix") else f"_{args.transform}"

    image_paths = find_images(images_dir)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for image_path in tqdm(image_paths, desc=f"Generating {args.transform}"):
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix(".txt")

        labels = read_yolo_labels(label_path)

        if args.skip_empty and not labels:
            continue

        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size

        transformed_image = transform.transform_image(image)

        transformed_labels = []

        for class_id, x_center, y_center, box_width, box_height in labels:
            bbox_xywh = yolo_to_coco_xywh(
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                image_width=original_width,
                image_height=original_height,
            )

            new_x, new_y, new_w, new_h = transform.transform_bbox_to_yolo(
                bbox_xywh=bbox_xywh,
                original_width=original_width,
                original_height=original_height,
            )

            if is_valid_yolo_box(new_x, new_y, new_w, new_h):
                transformed_labels.append((class_id, new_x, new_y, new_w, new_h))

        if args.skip_empty and not transformed_labels:
            continue

        output_image_name = f"{image_path.stem}{suffix}{image_path.suffix.lower()}"
        output_label_name = f"{image_path.stem}{suffix}.txt"

        output_image_path = output_images_dir / relative_path.parent / output_image_name
        output_label_path = output_labels_dir / relative_path.parent / output_label_name

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        transformed_image.save(output_image_path)

        write_yolo_labels(output_label_path, transformed_labels)

    print(f"Done. Transformed dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
