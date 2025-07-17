from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ConnectedComponent:
    """Dataclass to represent a connected component in a binary mask.

    A connected component is a set of pixels that are connected together.
    The bounding box is the smallest rectangle that contains all the pixels of the connected component.

    Attributes:
        mask: The binary mask of the connected component.
        bounding_box: The bounding box of the connected component. It is represented as a numpy array
        with the format [left, top, width, height].
    """

    mask: np.ndarray
    bounding_box: np.ndarray


def extract_connected_components_from_binary_mask(
    mask: np.ndarray, min_area: int = 1
) -> list[ConnectedComponent]:
    """Find the connected components and their corresponding bounding boxes in the binary masks.

    Args:
        mask: binary mask with shape (H, W, 1).
        min_area: The minimum area of a connected component to be considered valid.

    Returns:
        An list of ConnectedComponent corresponding to the valid connected component.
    """
    if (mask == 0).all():
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    valid_cc = []
    for label_idx in range(1, num_labels):  # Exclude background label (0)
        cc_stats = stats[label_idx]
        if cc_stats[cv2.CC_STAT_AREA] >= min_area:
            valid_cc.append(
                ConnectedComponent(
                    mask=(labels == label_idx).astype(np.uint8),
                    bounding_box=np.array(
                        (
                            cc_stats[cv2.CC_STAT_LEFT],
                            cc_stats[cv2.CC_STAT_TOP],
                            cc_stats[cv2.CC_STAT_WIDTH],
                            cc_stats[cv2.CC_STAT_HEIGHT],
                        ),
                        dtype=np.uint32,
                    ),
                )
            )

    return valid_cc
