import argparse
from pathlib import Path

from PIL import Image

from widefov.transforms.registry import available_transforms, get_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
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

    transform = get_transform(args.transform)
    transformed = transform.transform_image(image)

    transformed.save(output_path)
    print(f"Saved transformed image to: {output_path}")


if __name__ == "__main__":
    main()