import numpy as np


def choose_scale(width: int, height: int) -> float:
    """Choose resize scale used in the original RMFV365 scripts."""
    if height > 8000 or width > 8000:
        return 0.25
    if height > 4000 or width > 4000:
        return 0.5
    if height > 2000 or width > 2000:
        return 1.0
    if height > 1000 or width > 1000:
        return 2.0
    return 4.0


def compute_resized_shape(
    width: int,
    height: int,
    target_aspect_ratio: float = 3264 / 2448,
) -> tuple[int, int, float, float]:
    """
    Compute resized image shape following the original RMFV365 logic.

    Returns:
        new_width, new_height, scale_x, scale_y
    """
    scale = choose_scale(width, height)

    if width < height:
        new_width = int(width * scale)
        new_height = int(scale * int(width * target_aspect_ratio))
        scaling_new = int(new_width * target_aspect_ratio) / int(height * scale)

        scale_x = scale
        scale_y = scale * scaling_new
    else:
        new_width = int(scale * int(height * target_aspect_ratio))
        new_height = int(height * scale)
        scaling_new = int(new_height * target_aspect_ratio) / int(width * scale)

        scale_x = scale * scaling_new
        scale_y = scale

    return new_width, new_height, scale_x, scale_y


def crop_params_for_fisheye_independent(
    width: int,
    height: int,
    crop_ratio: float = 0.065,
) -> tuple[int, int, int, int]:
    """
    Crop parameters for the n=4/n=7 camera-independent fisheye transformation.

    Returns:
        left, top, right, bottom
    """
    left = int(crop_ratio * width) - 2
    top = int(crop_ratio * height) - 2
    right = width - int(crop_ratio * width) + 2
    bottom = height - int(crop_ratio * height) + 2
    return left, top, right, bottom


def clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip point coordinates to image boundaries."""
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points
