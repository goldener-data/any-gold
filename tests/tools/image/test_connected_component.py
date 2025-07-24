import numpy as np
import pytest
from any_gold.tools.image.connected_component import (
    extract_connected_components_from_binary_mask,
)


@pytest.fixture
def mask1() -> np.ndarray:
    """Fixture to provide a sample binary mask."""
    mask = np.zeros((10, 10, 1), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    return mask


@pytest.fixture
def mask2() -> np.ndarray:
    """Fixture to provide a sample binary mask."""
    mask = np.zeros((10, 10, 1), dtype=np.uint8)
    mask[6:9, 6:9, 0] = 1
    return mask


@pytest.fixture
def mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Fixture to provide a sample binary mask."""
    return mask1 + mask2


class TestExtractConnectedComponentsFromBinaryMask:
    def test_empty_mask(self):
        result = extract_connected_components_from_binary_mask(
            np.zeros((10, 10, 1), dtype=np.uint8)
        )
        assert result == []

    def test_multiple_components(
        self, mask1: np.ndarray, mask2: np.ndarray, mask: np.ndarray
    ):
        result = extract_connected_components_from_binary_mask(mask, min_area=1)
        assert len(result) == 2
        bboxes = [cc.bounding_box.tolist() for cc in result]
        assert result[0].area == 4
        assert result[1].area == 9
        assert [1, 1, 2, 2] == bboxes[0]
        assert [6, 6, 3, 3] == bboxes[1]
        assert np.array_equal(result[0].mask, mask1[..., 0])
        assert np.array_equal(result[1].mask, mask2[..., 0])

    def test_min_area(self, mask):
        mask = np.zeros((10, 10, 1), dtype=np.uint8)
        mask[0:2, 0:2, 0] = 1  # area 4
        mask[5:9, 5:9, 0] = 1  # area 16
        result = extract_connected_components_from_binary_mask(mask, min_area=10)
        assert len(result) == 1
        cc = result[0]
        assert np.array_equal(cc.bounding_box, np.array([5, 5, 4, 4], dtype=np.uint32))
