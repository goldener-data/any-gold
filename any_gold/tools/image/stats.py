from typing import Any

import numpy as np


def vision_segmentation_stats(
    shape_counts: dict[tuple[int, ...], int],
    areas: list[int],
    object_counts: list[int],
) -> dict[str, Any]:
    """Compute statistics for vision segmentation labels.

    Args:
        shape_counts: A dictionary where keys are shapes (tuples of dimensions) and values are counters of occurrences.
        areas: A list of areas for each connected component.
        object_counts: A list of object counts in each image.

    Returns:
        A dictionary containing:
        - 'shapes': A dictionary with the shape as key and the corresponding number of images in the dataset.
        - 'areas': A dictionary with min, max, mean, and total area, or None if empty.
        - 'object count': A dictionary with min, max, mean, and total object count, or None if empty.

    Raises:
        ValueError: If `shape_counts` is empty or if any shape has a count of 0.
    """
    if not shape_counts:
        raise ValueError("shape_counts must not be empty")

    shapes_stats = {}
    for shape, count in shape_counts.items():
        if count == 0:
            raise ValueError(f"Shape {shape} has a count of 0, which is invalid.")
        shapes_stats[shape] = count

    return {
        "shapes": shapes_stats,
        "areas": None
        if not areas
        else {
            "min": np.min(areas).item(),
            "max": np.max(areas).item(),
            "mean": np.mean(areas).item(),
            "total": np.sum(areas).item(),
        },
        "object count": None
        if not object_counts
        else {
            "min": np.min(object_counts).item(),
            "max": np.max(object_counts).item(),
            "mean": np.mean(object_counts).item(),
            "total": np.sum(object_counts).item(),
        },
    }
