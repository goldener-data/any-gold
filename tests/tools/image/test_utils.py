import pytest
import torch
from torchvision.tv_tensors import Image as TvImage, Mask as TvMask

from any_gold.tools.image.utils import (
    gold_segmentation_collate_fn,
    get_unique_pixel_values,
)
from any_gold.utils.dataset import AnyVisionSegmentationOutput


def _make_sample(
    index: int,
    label: str | set[str],
    image_shape: tuple[int, int, int] = (3, 4, 4),
    mask_shape: tuple[int, int, int] = (1, 4, 4),
) -> AnyVisionSegmentationOutput:
    image = TvImage(torch.zeros(image_shape, dtype=torch.uint8))
    mask = TvMask(torch.ones(mask_shape, dtype=torch.uint8))
    return AnyVisionSegmentationOutput(
        index=index,
        image=image,
        mask=mask,
        label=label,
    )


class TestGoldSegmentationCollateFn:
    def test_gold_segmentation_collate_fn_stacks_tensors_and_indices(self) -> None:
        batch = [
            _make_sample(index=0, label="cat"),
            _make_sample(index=1, label="dog"),
        ]

        output = gold_segmentation_collate_fn(batch)

        image = output["image"]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (2, 3, 4, 4)
        mask = output["mask"]
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (2, 1, 4, 4)
        label = output["label"]
        assert isinstance(label, list)
        assert label == ["cat", "dog"]
        output_index = output["index"]
        assert isinstance(output_index, torch.Tensor)
        assert torch.equal(output_index, torch.tensor([0, 1]))

    def test_gold_segmentation_collate_fn_keeps_list_labels(self) -> None:
        batch = [
            _make_sample(index=0, label=set(["cat", "pet"])),
            _make_sample(index=1, label=set(["dog", "pet"])),
        ]

        output = gold_segmentation_collate_fn(batch)

        assert output["label"] == [["cat", "pet"], ["dog", "pet"]]

    def test_gold_segmentation_collate_fn_raises_on_mismatched_shapes(self) -> None:
        batch = [
            _make_sample(index=0, label="cat", image_shape=(3, 4, 4)),
            _make_sample(index=1, label="dog", image_shape=(3, 5, 4)),
        ]

        with pytest.raises(RuntimeError):
            gold_segmentation_collate_fn(batch)


class TestGetUniquePixelValues:
    def test_single_channel_with_duplicates(self) -> None:
        tensor = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.uint8)
        result = get_unique_pixel_values(tensor)
        assert result == {
            (0,),
            (1,),
            (2,),
        }

    def test_rgb_tensor_with_duplicates(self) -> None:
        tensor = torch.zeros((3, 3, 3), dtype=torch.uint8)
        tensor[:, 0, 0] = torch.tensor([255, 0, 0])
        tensor[:, 1, 1] = torch.tensor([255, 0, 0])
        tensor[:, 2, 2] = torch.tensor([255, 0, 0])
        tensor[:, 0, 2] = torch.tensor([0, 0, 255])

        result = get_unique_pixel_values(tensor)
        assert result == {(0, 0, 0), (255, 0, 0), (0, 0, 255)}

    def test_raises_on_wrong_dimensions(self) -> None:
        # 2D tensor
        tensor_2d = torch.zeros((4, 4), dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected a tensor of shape"):
            get_unique_pixel_values(tensor_2d)

        # 4D tensor
        tensor_4d = torch.zeros((2, 3, 4, 4), dtype=torch.uint8)
        with pytest.raises(ValueError, match="Expected a tensor of shape"):
            get_unique_pixel_values(tensor_4d)
