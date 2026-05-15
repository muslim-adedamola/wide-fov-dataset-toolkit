import argparse
from pathlib import Path

from PIL import Image

from widefov.transforms.fisheye_independent import FisheyeIndependentTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--n", type=float, default=7.0, help="Fisheye scaling factor")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    transform = FisheyeIndependentTransform(n=args.n, suffix=f"_{int(args.n)}")
    transformed = transform.transform_image(image)

    transformed.save(output_path)
    print(f"Saved transformed image to: {output_path}")


if __name__ == "__main__":
    main()
