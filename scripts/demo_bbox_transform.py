import argparse

from widefov.transforms.registry import available_transforms, get_transform


def main():
    parser = argparse.ArgumentParser(
        description="Transform a single COCO-format bbox into YOLO format."
    )
    parser.add_argument("--width", type=int, required=True, help="Original image width")
    parser.add_argument("--height", type=int, required=True, help="Original image height")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("X", "Y", "W", "H"),
        help="COCO-format bbox: x y width height",
    )
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--transform",
        default="fisheye_n7",
        choices=available_transforms(),
        help="Transform to apply",
    )

    args = parser.parse_args()

    transform = get_transform(args.transform)

    x_center, y_center, box_width, box_height = transform.transform_bbox_to_yolo(
        bbox_xywh=tuple(args.bbox),
        original_width=args.width,
        original_height=args.height,
    )

    print("Transformed YOLO label:")
    print(
        f"{args.class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}"
    )


if __name__ == "__main__":
    main()