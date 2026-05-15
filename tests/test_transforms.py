from pathlib import Path

import numpy as np
from PIL import Image

from widefov.transforms.registry import available_transforms, get_transform


def make_dummy_image(width=640, height=480):
    """Create a simple RGB image for transform smoke tests."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[100:300, 150:400, :] = 255
    return Image.fromarray(image)


def test_all_transforms_run_on_image():
    image = make_dummy_image()

    for transform_name in available_transforms():
        transform = get_transform(transform_name)
        output = transform.transform_image(image)

        assert isinstance(output, Image.Image)
        assert output.width > 0
        assert output.height > 0


def test_all_transforms_return_valid_bbox_tuple():
    image = make_dummy_image()
    original_width, original_height = image.size
    bbox = (100, 80, 200, 150)

    for transform_name in available_transforms():
        transform = get_transform(transform_name)

        yolo_box = transform.transform_bbox_to_yolo(
            bbox_xywh=bbox,
            original_width=original_width,
            original_height=original_height,
        )

        assert len(yolo_box) == 4

        x_center, y_center, width, height = yolo_box

        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert width > 0
        assert height > 0
