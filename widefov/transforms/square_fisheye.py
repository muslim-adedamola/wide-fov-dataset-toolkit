from dataclasses import dataclass

import numpy as np
from PIL import Image

from widefov.annotations.bbox import points_to_xyxy, xywh_to_points, xyxy_to_yolo
from widefov.transforms.common import compute_resized_shape


def crop_params_for_square_fisheye(
    square_size: int,
    original_width: int,
    original_height: int,
    crop_c: float = 0.11,
    crop_b: float = 0.17,
) -> tuple[int, int, int, int]:
    """
    Crop parameters for the square fisheye variant.

    Follows the original RMFV365 square script:
    - portrait images use b horizontally and c vertically
    - landscape/square images use c horizontally and b vertically
    """
    width = square_size
    height = square_size

    if original_height > original_width:
        left = int(crop_b * width) - 10
        top = int(crop_c * height) - 10
        right = width - int(crop_b * width) + 10
        bottom = height - int(crop_c * height) + 10
    else:
        left = int(crop_c * width) - 10
        top = int(crop_b * height) - 10
        right = width - int(crop_c * width) + 10
        bottom = height - int(crop_b * height) + 10

    return left, top, right, bottom


@dataclass
class SquareFisheyeTransform:
    """
    Square-canvas camera/lens-independent fisheye transform.

    This implements the RMFV365 square variant:
    resize image, place it at the center of a square canvas,
    apply the n=4 fisheye-independent mapping, then crop.
    """

    n: float = 4.0
    target_aspect_ratio: float = 3264 / 2448
    crop_c: float = 0.11
    crop_b: float = 0.17
    suffix: str = "_square"

    def _prepare_square_image(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, int, int, int, int]:
        """
        Resize image and place it on a square canvas.

        Returns:
            square_image, resized_width, resized_height, offset_x, offset_y
        """
        original_width, original_height = image.size

        resized_width, resized_height, _, _ = compute_resized_shape(
            width=original_width,
            height=original_height,
            target_aspect_ratio=self.target_aspect_ratio,
        )

        resized = image.resize((resized_width, resized_height), Image.BICUBIC)
        square_size = max(resized_width, resized_height)

        if resized.mode == "RGB":
            canvas = Image.new("RGB", (square_size, square_size), (0, 0, 0))
        else:
            canvas = Image.new(resized.mode, (square_size, square_size), 0)

        offset_x = 0
        offset_y = 0

        if resized_height < resized_width:
            offset_y = int((resized_width - resized_height) / 2)
        elif resized_height > resized_width:
            offset_x = int((resized_height - resized_width) / 2)

        canvas.paste(resized, (offset_x, offset_y))

        return canvas, resized_width, resized_height, offset_x, offset_y

    def transform_points(
        self,
        points: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        resized_width, resized_height, scale_x, scale_y = compute_resized_shape(
            width=original_width,
            height=original_height,
            target_aspect_ratio=self.target_aspect_ratio,
        )

        square_size = max(resized_width, resized_height)

        offset_x = 0
        offset_y = 0

        if resized_height < resized_width:
            offset_y = int((resized_width - resized_height) / 2)
        elif resized_height > resized_width:
            offset_x = int((resized_height - resized_width) / 2)

        points = points.astype(np.float32).copy()
        points[:, 0] = points[:, 0] * scale_x + offset_x
        points[:, 1] = points[:, 1] * scale_y + offset_y

        x = 2 * points[:, 0] / square_size - 1
        y = 2 * points[:, 1] / square_size - 1

        x1 = x * np.sqrt(1 - y**2 / 2)
        y1 = y * np.sqrt(1 - x**2 / 2)

        r = np.sqrt(x1**2 + y1**2)

        x2 = x1 * np.exp(-(r**2) / self.n)
        y2 = y1 * np.exp(-(r**2) / self.n)

        transformed_x = square_size * (x2 + 1) / 2
        transformed_y = square_size * (y2 + 1) / 2

        transformed_points = np.stack([transformed_x, transformed_y], axis=1)

        crop_box = crop_params_for_square_fisheye(
            square_size=square_size,
            original_width=original_width,
            original_height=original_height,
            crop_c=self.crop_c,
            crop_b=self.crop_b,
        )

        left, top, _, _ = crop_box
        transformed_points[:, 0] -= left
        transformed_points[:, 1] -= top

        return transformed_points, crop_box

    def transform_image(self, image: Image.Image) -> Image.Image:
        original_width, original_height = image.size

        square_image, _, _, _, _ = self._prepare_square_image(image)
        width, height = square_image.size

        image_array = np.asarray(square_image)

        if square_image.mode == "RGB":
            transformed = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            transformed = np.zeros((height, width), dtype=np.uint8)

        y_grid = np.arange(height)
        x_grid = np.arange(width)
        xi, yi = np.meshgrid(x_grid, y_grid)

        x = 2 * xi / width - 1
        y = 2 * yi / height - 1

        x1 = x * np.sqrt(1 - y**2 / 2)
        y1 = y * np.sqrt(1 - x**2 / 2)

        r = np.sqrt(x1**2 + y1**2)

        x2 = x1 * np.exp(-(r**2) / self.n)
        y2 = y1 * np.exp(-(r**2) / self.n)

        xf = (width * (x2 + 1) / 2).astype(np.int32)
        yf = (height * (y2 + 1) / 2).astype(np.int32)

        valid = (xf >= 0) & (xf < width) & (yf >= 0) & (yf < height)
        transformed[yf[valid], xf[valid]] = image_array[yi[valid], xi[valid]]

        transformed_image = Image.fromarray(transformed)

        crop_box = crop_params_for_square_fisheye(
            square_size=width,
            original_width=original_width,
            original_height=original_height,
            crop_c=self.crop_c,
            crop_b=self.crop_b,
        )

        return transformed_image.crop(crop_box)

    def transform_bbox_to_yolo(
        self,
        bbox_xywh: tuple[float, float, float, float],
        original_width: int,
        original_height: int,
    ) -> tuple[float, float, float, float]:
        x, y, w, h = bbox_xywh
        points = xywh_to_points(x, y, w, h)

        transformed_points, crop_box = self.transform_points(
            points=points,
            original_width=original_width,
            original_height=original_height,
        )

        left, top, right, bottom = crop_box
        cropped_width = right - left
        cropped_height = bottom - top

        x_min, y_min, x_max, y_max = points_to_xyxy(transformed_points)

        return xyxy_to_yolo(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            image_width=cropped_width,
            image_height=cropped_height,
        )
