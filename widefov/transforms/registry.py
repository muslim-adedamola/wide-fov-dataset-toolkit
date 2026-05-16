from widefov.transforms.barrel import BarrelTransform
from widefov.transforms.division import DivisionModelTransform
from widefov.transforms.fisheye_independent import FisheyeIndependentTransform
from widefov.transforms.equidistance import EquidistanceProjectionTransform
from widefov.transforms.square_fisheye import SquareFisheyeTransform


def get_transform(name: str):
    """
    Build a transform by name.

    Supported names:
    - fisheye_n4
    - fisheye_n7
    - barrel1
    - barrel2
    - division1
    - division2
    - square_fisheye
    - equidistance
    """
    name = name.lower()

    if name == "fisheye_n4":
        return FisheyeIndependentTransform(
            n=4.0,
            crop_ratio=0.11,
            suffix="_4",
        )

    if name == "fisheye_n7":
        return FisheyeIndependentTransform(
            n=7.0,
            crop_ratio=0.065,
            suffix="_7",
        )

    if name == "barrel1":
        return BarrelTransform(
            distortion_coeff=0.25,
            crop_a=0.08,
            crop_b=0.045,
            suffix="_barrel1",
        )

    if name == "barrel2":
        return BarrelTransform(
            distortion_coeff=0.125,
            crop_a=0.04,
            crop_b=0.02,
            suffix="_barrel2",
        )

    if name == "division1":
        return DivisionModelTransform(
            distortion_amount=0.38,
            crop_a=0.155,
            crop_b=0.11,
            suffix="_div1",
        )

    if name == "division2":
        return DivisionModelTransform(
            distortion_amount=0.55,
            crop_a=0.24,
            crop_b=0.19,
            suffix="_div2",
        )
    
    if name == "square_fisheye":
        return SquareFisheyeTransform(
            n=4.0,
            crop_c=0.11,
            crop_b=0.17,
            suffix="_square",
        )

    if name == "equidistance":
        return EquidistanceProjectionTransform(
            focal_length=1000.0,
            crop_a=0.185,
            crop_b=0.14,
            suffix="_f",
        )

    raise ValueError(
        f"Unknown transform '{name}'. "
        "Available transforms: "
        "fisheye_n4, fisheye_n7, barrel1, barrel2, division1, division2, square_fisheye, equidistance"
    )


def available_transforms() -> list[str]:
    return [
        "fisheye_n4",
        "fisheye_n7",
        "barrel1",
        "barrel2",
        "division1",
        "division2",
        "square_fisheye",
        "equidistance",
    ]
