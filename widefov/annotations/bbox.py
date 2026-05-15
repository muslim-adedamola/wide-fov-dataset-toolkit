import numpy as np


def xywh_to_points(x: float, y: float, w: float, h: float) -> np.ndarray:
    """
    Convert a bbox in COCO xywh format to 8 representative points.

    Points include:
    - four corners
    - four edge midpoints

    This follows the original RMFV365 coordinate transformation strategy.
    """
    xtl = x
    ytl = y
    xbr = x + w
    ybr = y + h

    xtr = xbr
    ytr = ytl

    xbl = xtl
    ybl = ybr

    top_mid = [(xtr + xtl) / 2, (ytr + ytl) / 2]
    left_mid = [(xbl + xtl) / 2, (ybl + ytl) / 2]
    right_mid = [(xbr + xtr) / 2, (ybr + ytr) / 2]
    bottom_mid = [(xbr + xbl) / 2, (ybr + ybl) / 2]

    return np.array(
        [
            [xtl, ytl],
            [xbr, ybr],
            [xtr, ytr],
            [xbl, ybl],
            top_mid,
            left_mid,
            right_mid,
            bottom_mid,
        ],
        dtype=np.float32,
    )


def points_to_xyxy(points: np.ndarray) -> tuple[float, float, float, float]:
    """Create an axis-aligned bbox from transformed points."""
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    return x_min, y_min, x_max, y_max


def xyxy_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert xyxy bbox to normalized YOLO xywh format."""
    box_width = max(0.0, x_max - x_min)
    box_height = max(0.0, y_max - y_min)

    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def is_valid_yolo_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    min_size: float = 1e-6,
) -> bool:
    """Check if a YOLO bbox is valid after transformation."""
    if width <= min_size or height <= min_size:
        return False

    if x_center < 0 or x_center > 1:
        return False

    if y_center < 0 or y_center > 1:
        return False

    return True
