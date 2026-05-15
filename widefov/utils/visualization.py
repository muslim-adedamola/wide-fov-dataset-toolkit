from PIL import Image, ImageDraw


def draw_xyxy_box(
    image: Image.Image,
    box: tuple[float, float, float, float],
    label: str | None = None,
    width: int = 3,
) -> Image.Image:
    """Draw an xyxy bounding box on a copy of an image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)

    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=width)

    if label:
        draw.text((x_min, max(0, y_min - 12)), label, fill="red")

    return image


def make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Create a side-by-side image after matching heights."""
    left = left.convert("RGB")
    right = right.convert("RGB")

    target_height = max(left.height, right.height)

    def resize_to_height(image: Image.Image, height: int) -> Image.Image:
        if image.height == height:
            return image
        scale = height / image.height
        new_width = int(image.width * scale)
        return image.resize((new_width, height), Image.BICUBIC)

    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    canvas = Image.new("RGB", (left.width + right.width, target_height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    return canvas
