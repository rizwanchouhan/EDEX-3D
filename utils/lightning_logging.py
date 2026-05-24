from skimage.io import imsave
from pathlib import Path
from wandb import Image  # Importing Image from wandb package
import numpy as np


def _fix_image(image):
    """
    Fix the image intensity values and clip them to the valid range.
    Args:
        image: Input image.

    Returns:
        Fixed image.
    """
    if image.max() < 30.:
        image = image * 255.
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _log_wandb_image(path, image, caption=None):
    """
    Log image to Weights & Biases (wandb).
    Args:
        path: Path to save the image.
        image: Image data.
        caption: Caption for the image (default=None).

    Returns:
        Wandb Image object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _fix_image(image)
    imsave(path, image)
    if caption is not None:
        caption_file = Path(path).parent / (Path(path).stem + ".txt")
        with open(caption_file, "w") as f:
            f.write(caption)
    wandb_image = Image(str(path), caption=caption)
    return wandb_image


def _log_array_image(path, image, caption=None):
    """
    Log array image.
    Args:
        path: Path to save the image.
        image: Image data.
        caption: Caption for the image (default=None).

    Returns:
        Processed image.
    """
    image = _fix_image(image)
    if path is not None:
        imsave(path, image)
    return image


def _torch_image2np(torch_image):
    """
    Convert torch tensor image to numpy array.
    Args:
        torch_image: Torch tensor image.

    Returns:
        Numpy array image.
    """
    image = torch_image.detach().cpu().numpy()
    if len(image.shape) == 4:
        image = image.transpose([0, 2, 3, 1])
    elif len(image.shape) == 3:
        image = image.transpose([1, 2, 0])
    return image
