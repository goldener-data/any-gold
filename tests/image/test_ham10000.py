import pytest


from any_gold import HAM10000Dataset
from tests.conftest import TEST_DATASET_LOADING


class TestHAM10000Dataset:
    def test_handle(self):
        assert hasattr(HAM10000Dataset, "_HANDLE")

    @pytest.mark.skipif(
        not TEST_DATASET_LOADING, reason="TEST_DATASET_LOADING is not True"
    )
    def test_dataset(self):
        dataset = HAM10000Dataset(
            root="/Users/boudoul/Desktop/dev/datasets/1", split="part1"
        )
        assert len(dataset) > 100
        output = dataset[0]
        assert "image" in output
        assert "label" in output
