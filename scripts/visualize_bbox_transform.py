import argparse
from pathlib import Path

from PIL import Image

from widefov.annotations.bbox import xywh_to_points, points_to_xyxy
from widefov.transforms.registry import available_transforms, get_transform
from widefov.utils.visualization import draw_xyxy_box, make_side_by_side


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="COCO-format bbox: x y width height",
    )
    parser.add_argument("--output", required=True, help="Path to output debug image")
    parser.add_argument(
        "--transform",
        default="fisheye_n7",
        choices=available_transforms(),
        help="Transform to apply",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    transform = get_transform(args.transform)

    x, y, w, h = args.bbox
    original_box_xyxy = (x, y, x + w, y + h)

    original_debug = draw_xyxy_box(image, original_box_xyxy, label="original")

    transformed_image = transform.transform_image(image)

    points = xywh_to_points(x, y, w, h)
    transformed_points, _ = transform.transform_points(
        points=points,
        original_width=original_width,
        original_height=original_height,
    )
    transformed_box_xyxy = points_to_xyxy(transformed_points)

    transformed_debug = draw_xyxy_box(
        transformed_image,
        transformed_box_xyxy,
        label="transformed",
    )

    side_by_side = make_side_by_side(original_debug, transformed_debug)
    side_by_side.save(output_path)

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()