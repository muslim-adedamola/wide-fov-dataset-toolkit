from dataclasses import dataclass

import numpy as np
from PIL import Image

from widefov.annotations.bbox import points_to_xyxy, xywh_to_points, xyxy_to_yolo
from widefov.transforms.common import (
    compute_resized_shape,
    crop_params_for_fisheye_independent,
)


@dataclass
class FisheyeIndependentTransform:
    """
    Camera/lens-independent fisheye-style transformation.

    This implements the transformation used for the RMFV365 n=4 and n=7 variants.
    """

    n: float = 7.0
    target_aspect_ratio: float = 3264 / 2448
    crop_ratio: float = 0.065
    suffix: str = "_7"

    def resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        new_width, new_height, _, _ = compute_resized_shape(
            width=width,
            height=height,
            target_aspect_ratio=self.target_aspect_ratio,
        )
        return image.resize((new_width, new_height), Image.BICUBIC)

    def transform_points(
        self,
        points: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Transform points from the original image coordinate system.

        Returns:
            transformed_points_after_crop, crop_box
        """
        resized_width, resized_height, scale_x, scale_y = compute_resized_shape(
            width=original_width,
            height=original_height,
            target_aspect_ratio=self.target_aspect_ratio,
        )

        points = points.astype(np.float32).copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y

        x = 2 * points[:, 0] / resized_width - 1
        y = 2 * points[:, 1] / resized_height - 1

        x1 = x * np.sqrt(1 - y**2 / 2)
        y1 = y * np.sqrt(1 - x**2 / 2)

        r = np.sqrt(x1**2 + y1**2)

        x2 = x1 * np.exp(-(r**2) / self.n)
        y2 = y1 * np.exp(-(r**2) / self.n)

        transformed_x = resized_width * (x2 + 1) / 2
        transformed_y = resized_height * (y2 + 1) / 2

        transformed_points = np.stack([transformed_x, transformed_y], axis=1)

        left, top, right, bottom = crop_params_for_fisheye_independent(
            width=resized_width,
            height=resized_height,
            crop_ratio=self.crop_ratio,
        )

        transformed_points[:, 0] -= left
        transformed_points[:, 1] -= top

        return transformed_points, (left, top, right, bottom)

    def transform_image(self, image: Image.Image) -> Image.Image:
        """
        Transform a PIL image and crop black borders.

        This follows the original forward-mapping implementation used in RMFV365.
        """
        image = self.resize_image(image)
        width, height = image.size

        image_array = np.asarray(image)

        if image.mode == "RGB":
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

        crop_box = crop_params_for_fisheye_independent(
            width=width,
            height=height,
            crop_ratio=self.crop_ratio,
        )

        return transformed_image.crop(crop_box)

    def transform_bbox_to_yolo(
        self,
        bbox_xywh: tuple[float, float, float, float],
        original_width: int,
        original_height: int,
    ) -> tuple[float, float, float, float]:
        """
        Transform a COCO-format bbox into normalized YOLO format.
        """
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
