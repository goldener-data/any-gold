from pathlib import Path

from torchvision.tv_tensors import Image as TvImage, Mask as TvMask
from PIL import Image as PILImage


def load_torchvision_image(path: Path) -> TvImage:
    pil_image = PILImage.open(path).convert("RGB")
    return TvImage(pil_image).unsqueeze(0)


def load_torchvision_mask(path: Path) -> TvMask:
    pil_mask = PILImage.open(path).convert("L")
    return TvMask(pil_mask).unsqueeze(0)
