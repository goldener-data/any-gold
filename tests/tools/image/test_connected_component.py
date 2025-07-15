import numpy as np
import pytest
from any_gold.tools.image.connected_component import (
    extract_connected_components_from_binary_mask,
)


@pytest.fixture
def mask():
    """Fixture to provide a sample binary mask."""
    mask = np.zeros((10, 10, 1), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    mask[6:8, 6:9, 0] = 1
    return mask


class TestExtractConnectedComponentsFromBinaryMask:
    def test_empty_mask(self):
        result = extract_connected_components_from_binary_mask(
            np.zeros((10, 10, 1), dtype=np.uint8)
        )
        assert result == []

    def test_multiple_components(self, mask):
        result = extract_connected_components_from_binary_mask(mask, min_area=1)
        assert len(result) == 2
        bboxes = [cc.bounding_box.tolist() for cc in result]
        assert [1, 1, 2, 2] in bboxes
        assert [6, 6, 3, 2] in bboxes

    def test_min_area(self, mask):
        mask = np.zeros((10, 10, 1), dtype=np.uint8)
        mask[0:2, 0:2, 0] = 1  # area 4
        mask[5:9, 5:9, 0] = 1  # area 16
        result = extract_connected_components_from_binary_mask(mask, min_area=10)
        assert len(result) == 1
        cc = result[0]
        assert np.array_equal(cc.bounding_box, np.array([5, 5, 4, 4], dtype=np.uint32))
