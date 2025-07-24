from typing import Dict, List, Tuple
import pytest
import numpy as np
from any_gold.tools.image.stats import vision_segmentation_stats


class TestVisionSegmentationStats:
    def test_shapes_and_areas_and_object_counts(self) -> None:
        shape_counts: Dict[Tuple[int, ...], int] = {(64, 64): 2, (32, 32): 1}
        areas: List[int] = [100, 200, 300]
        object_counts: List[int] = [1, 2, 3]
        result = vision_segmentation_stats(shape_counts, areas, object_counts)
        assert result["shapes"] == shape_counts
        assert result["areas"]["min"] == 100
        assert result["areas"]["max"] == 300
        assert result["areas"]["mean"] == pytest.approx(np.mean(areas))
        assert result["areas"]["total"] == 600
        assert result["object count"]["min"] == 1
        assert result["object count"]["max"] == 3
        assert result["object count"]["mean"] == pytest.approx(np.mean(object_counts))
        assert result["object count"]["total"] == 6

    def test_empty_areas_and_object_counts(self) -> None:
        shape_counts: Dict[Tuple[int, ...], int] = {(64, 64): 1}
        areas: List[int] = []
        object_counts: List[int] = []
        result = vision_segmentation_stats(shape_counts, areas, object_counts)
        assert result["areas"] is None
        assert result["object count"] is None

    def test_empty_shape_counts(self) -> None:
        shape_counts: Dict[Tuple[int, ...], int] = {}
        areas: List[int] = [10]
        object_counts: List[int] = [1]
        with pytest.raises(ValueError):
            vision_segmentation_stats(shape_counts, areas, object_counts)

    def test_zero_shape_count(self) -> None:
        shape_counts: Dict[Tuple[int, ...], int] = {(64, 64): 0}
        areas: List[int] = [10]
        object_counts: List[int] = [1]
        with pytest.raises(ValueError):
            vision_segmentation_stats(shape_counts, areas, object_counts)
