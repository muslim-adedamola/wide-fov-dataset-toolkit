from dataclasses import dataclass

import numpy as np
from PIL import Image

from widefov.annotations.bbox import points_to_xyxy, xywh_to_points, xyxy_to_yolo


def equidistance_output_shape(
    original_width: int,
    original_height: int,
    landscape_width: int = 3264,
    landscape_height: int = 2448,
) -> tuple[int, int]:
    """
    Return the resized shape used by the original RMFV365 equidistance script.
    """
    if original_width < original_height:
        return landscape_height, landscape_width

    return landscape_width, landscape_height


def crop_params_for_equidistance(
    width: int,
    height: int,
    crop_a: float = 0.185,
    crop_b: float = 0.14,
) -> tuple[int, int, int, int]:
    """
    Crop parameters for equidistance projection.

    Follows the original RMFV365 equidistance script.
    """
    if height > width:
        left = int(crop_b * width) - 10
        top = int(crop_a * height) - 10
        right = width - int(crop_b * width) + 10
        bottom = height - int(crop_a * height) + 10
    else:
        left = int(crop_a * width) - 10
        top = int(crop_b * height) - 10
        right = width - int(crop_a * width) + 10
        bottom = height - int(crop_b * height) + 10

    return left, top, right, bottom


@dataclass
class EquidistanceProjectionTransform:
    """
    Equidistance projection transform.

    This implements the RMFV365 equidistance projection variant.
    The original setting used focal_length=1000 and suffix "_f".
    """

    focal_length: float = 1000.0
    landscape_width: int = 3264
    landscape_height: int = 2448
    crop_a: float = 0.185
    crop_b: float = 0.14
    suffix: str = "_f"

    def resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        new_width, new_height = equidistance_output_shape(
            original_width=width,
            original_height=height,
            landscape_width=self.landscape_width,
            landscape_height=self.landscape_height,
        )
        return image.resize((new_width, new_height), Image.BICUBIC)

    def transform_points(
        self,
        points: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        resized_width, resized_height = equidistance_output_shape(
            original_width=original_width,
            original_height=original_height,
            landscape_width=self.landscape_width,
            landscape_height=self.landscape_height,
        )

        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        points = points.astype(np.float32).copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y

        x = points[:, 0] - resized_width / 2
        y = points[:, 1] - resized_height / 2

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        s1 = self.focal_length * np.arctan2(r, self.focal_length)

        transformed_x = s1 * np.cos(theta) + resized_width / 2
        transformed_y = s1 * np.sin(theta) + resized_height / 2

        transformed_points = np.stack([transformed_x, transformed_y], axis=1)

        crop_box = crop_params_for_equidistance(
            width=resized_width,
            height=resized_height,
            crop_a=self.crop_a,
            crop_b=self.crop_b,
        )

        left, top, _, _ = crop_box
        transformed_points[:, 0] -= left
        transformed_points[:, 1] -= top

        return transformed_points, crop_box

    def transform_image(self, image: Image.Image) -> Image.Image:
        image = self.resize_image(image)
        width, height = image.size

        image_array = np.asarray(image)

        y_grid = np.arange(height)
        x_grid = np.arange(width)
        xi, yi = np.meshgrid(x_grid, y_grid)

        xt = xi - width / 2
        yt = yi - height / 2

        r = np.sqrt(xt**2 + yt**2)
        theta = np.arctan2(yt, xt)

        s1 = self.focal_length * np.arctan2(r, self.focal_length)

        x2 = s1 * np.cos(theta)
        y2 = s1 * np.sin(theta)

        xf = (x2 + width / 2).astype(np.int32)
        yf = (y2 + height / 2).astype(np.int32)

        if image.mode == "RGB":
            transformed = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            transformed = np.zeros((height, width), dtype=np.uint8)

        valid = (xf >= 0) & (xf < width) & (yf >= 0) & (yf < height)
        transformed[yf[valid], xf[valid]] = image_array[yi[valid], xi[valid]]

        transformed_image = Image.fromarray(transformed)

        crop_box = crop_params_for_equidistance(
            width=width,
            height=height,
            crop_a=self.crop_a,
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
