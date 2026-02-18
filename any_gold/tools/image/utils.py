import torch

from any_gold.utils.dataset import AnyVisionSegmentationOutput


def gold_segmentation_collate_fn(
    batch: list[AnyVisionSegmentationOutput],
) -> dict[str, torch.Tensor | list[str | set[str]]]:
    """Collate function for gold vision segmentation dataset.

    This collate function will stack the images, masks and index in the batch.
    The labels will be kept as a list.

    Args:
        batch: A list of AnyVisionSegmentationOutput objects.

    Returns:
        A dictionary containing the stacked images, masks, labels and indices.
    """
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": [item["label"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch]),
    }


def get_unique_pixel_values(tensor: torch.Tensor) -> set[tuple[int]]:
    """Get the set of channel values in location of a tensor.

    Args:
        tensor: A tensor of shape (C, H, W).

    Returns:
        A set of tuples, where each tuple represents the unique channel values
        at a location in the tensor.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a tensor of shape (C, H, W), but got {tensor.shape}"
        )

    flattened = tensor.moveaxis(0, -1).flatten(
        0,
        1,
    )
    unique_list = torch.unique(flattened, dim=0, return_counts=False).tolist()
    return set(tuple(value) for value in unique_list)
