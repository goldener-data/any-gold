from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
from torch import Generator
from torch.utils.data import RandomSampler, DataLoader

import fiftyone as fo

from any_gold.tools.image.connected_component import (
    ConnectedComponent,
    extract_connected_components_from_binary_mask,
)
from any_gold.utils.dataset import (
    AnyVisionSegmentationOutput,
    AnyVisionSegmentationDataset,
)


def build_fo_detections_from_connected_components(
    connected_components: list[ConnectedComponent],
    label: str,
    index: int,
    save_dir: Path,
) -> fo.Detections:
    """Build FiftyOne detections from connected components.

    Args:
        connected_components: List of connected components.
        label: The label for the detections.
        index: The index of the sample.
        save_dir: Directory to save the masks. It is more efficient to save them locally than storing everything in RAM.
        If the data is not required to be kept, it can be a temporary directory.

    Returns:
        FiftyOne detections including all connected components.

    Raises:
        ValueError: If no connected components are provided.
    """
    if not connected_components:
        raise ValueError("No connected components provided to build detections.")

    detections = []
    for cc_idx, cc in enumerate(connected_components):
        mask_h, mask_w = cc.mask.shape[:2]
        bb_left, bb_top, bb_w, bb_h = cc.bounding_box
        cc_mask = cc.mask[
            bb_top : bb_top + bb_h, bb_left : bb_left + bb_w
        ]  # mask must only correspond to the bounding box
        mask_path = str(save_dir / f"{label}_{index}_{cc_idx}.png")
        cv2.imwrite(mask_path, cc_mask)

        detections.append(
            fo.Detection(
                label=label,
                bounding_box=cc.bounding_box
                / np.array(
                    [mask_w, mask_h, mask_w, mask_h]
                ),  # fiftyone expects bounding box normalized to [0, 1] from width and height values
                mask_path=mask_path,
            )
        )

    return fo.Detections(detections=detections)


def build_fo_sample_from_any_vision_segmentation_output(
    sample: AnyVisionSegmentationOutput,
    save_dir: Path,
) -> fo.Sample:
    """Build a FiftyOne sample from an AnyVisionSegmentationOutput.

    So far, it is assumed that the mask provided by an AnyVisionSegmentationOutput is a binary masks.

    Args:
        sample: The AnyVisionSegmentationOutput to convert.
        save_dir: The directory where the images and masks will be saved. It is more efficient to save them locally
        than storing everything in RAM. If the data is not required to be kept, it can be a temporary directory.

    Returns:
        A FiftyOne sample containing the image, mask, and metadata.
    """
    label = sample["label"]
    index = sample["index"]

    # save the image locally
    image = sample["image"].permute(1, 2, 0).numpy()
    image_path = save_dir / f"image_{index}.png"
    cv2.imwrite(str(image_path), image)

    # save the masks locally and create fiftyone detections
    cc_masks_and_boxes = extract_connected_components_from_binary_mask(
        sample["mask"].permute(1, 2, 0).numpy()
    )
    detections = (
        build_fo_detections_from_connected_components(
            cc_masks_and_boxes,
            label=label,
            index=index,
            save_dir=save_dir,
        )
        if cc_masks_and_boxes
        else None
    )

    return fo.Sample(
        filepath=str(image_path),
        ground_truth=detections,
        metadata=fo.ImageMetadata(
            width=image.shape[1],
            height=image.shape[0],
            num_channels=image.shape[2] if len(image.shape) == 3 else 1,
            index=index,
        ),
    )


def build_fo_dataset_from_any_vision_segmentation_dataset(
    dataset: AnyVisionSegmentationDataset,
    dataset_name: str,
    save_dir: Path,
    num_samples: int | None = None,
    persistent: bool = False,
    overwrite: bool = True,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int | None = None,
) -> fo.Dataset:
    """Build a FiftyOne dataset from an AnyVisionSegmentationDataset.

    Args:
        dataset: The AnyVisionSegmentationDataset to convert. It is assumed that the segmentation masks are binary masks.
        dataset_name: The name of the FiftyOne dataset.
        save_dir: The directory where the images and masks will be saved.
        num_samples: The number of samples to include in the dataset. If None, all samples will be included.
        persistent: If True, the dataset will be persistent in fiftyone and it will not be possible to create
        a dataset with same name without setting `overwrite` to True.
        overwrite: If True, the existing dataset will be overwritten.
        batch_size: The batch size for the DataLoader.
        num_workers: The number of workers for the DataLoader.
        seed: The random seed for sampling the dataset. If None, no specific seed is set.

    Returns:
        A FiftyOne dataset containing the samples from the AnyVisionSegmentationDataset.
    """
    fo_dataset = fo.Dataset(dataset_name, persistent=persistent, overwrite=overwrite)

    random_generator = Generator().manual_seed(seed) if seed is not None else None
    sampler = (
        RandomSampler(
            dataset,
            replacement=False,
            num_samples=num_samples,
            generator=random_generator,
        )
        if num_samples is not None
        else None
    )
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # get directly access to the samples without collating
    )

    for batch in tqdm(dataloader, desc="Converting samples to FiftyOne format"):
        for sample in batch:
            fo_dataset.add_sample(
                build_fo_sample_from_any_vision_segmentation_output(
                    sample=sample,
                    save_dir=save_dir,
                )
            )

    return fo_dataset
