import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask
from any_gold.utils.dataset import (
    AnyVisionSegmentationDataset,
    AnyVisionSegmentationOutput,
)


class DummyDataset(AnyVisionSegmentationDataset):
    def __init__(self) -> None:
        super().__init__(root="/tmp")
        image = torch.ones((3, 4, 4), dtype=torch.uint8)
        mask = torch.zeros((1, 4, 4), dtype=torch.uint8)
        mask[0, 1:3, 1:3] = 1

        self._data = [
            AnyVisionSegmentationOutput(
                index=0,
                image=TvImage(image),
                mask=TvMask(mask),
                label="dog",
            ),
            AnyVisionSegmentationOutput(
                index=1,
                image=TvImage(image),
                mask=TvMask(mask),
                label="cat",
            ),
        ]

    def get_raw(self, index: int) -> AnyVisionSegmentationOutput:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __get_item__(self, index: int) -> AnyVisionSegmentationOutput:
        return self.get_raw(index)


class TestAnyVisionSegmentationDatasetDescribe:
    def test_describe(self) -> None:
        dataset = DummyDataset()
        description = dataset.describe()
        assert isinstance(description, dict)
        assert description["name"] == "DummyDataset"
        assert description["sample count"] == 2
        assert description["shapes"] == {(3, 4, 4): 2}
        assert description["areas"]["min"] == 4
        assert description["areas"]["max"] == 4
        assert description["areas"]["mean"] == 4.0
        assert description["areas"]["total"] == 8
        assert description["object count"]["min"] == 1
        assert description["object count"]["max"] == 1
        assert description["object count"]["mean"] == 1.0
        assert description["object count"]["total"] == 2
        assert isinstance(description["cat"], dict)
        assert description["cat"]["areas"]["min"] == 4
        assert description["cat"]["areas"]["max"] == 4
        assert description["cat"]["areas"]["mean"] == 4.0
        assert description["cat"]["areas"]["total"] == 4
        assert description["cat"]["object count"]["min"] == 1
        assert description["cat"]["object count"]["max"] == 1
        assert description["cat"]["object count"]["mean"] == 1.0
        assert description["cat"]["object count"]["total"] == 1
        assert isinstance(description["dog"], dict)
        assert description["dog"]["areas"]["min"] == 4
        assert description["dog"]["areas"]["max"] == 4
        assert description["dog"]["areas"]["mean"] == 4.0
        assert description["dog"]["areas"]["total"] == 4
        assert description["dog"]["object count"]["min"] == 1
        assert description["dog"]["object count"]["max"] == 1
        assert description["dog"]["object count"]["mean"] == 1.0
        assert description["dog"]["object count"]["total"] == 1
