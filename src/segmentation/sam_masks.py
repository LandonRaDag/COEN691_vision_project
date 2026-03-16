"""
SAM-based 2D segmentation utilities.

This module handles generating object masks from images using
the Segment Anything Model (SAM).
"""

from typing import List
import numpy as np


def generate_masks(image_dir: str, output_dir: str) -> List[np.ndarray]:
    """
    Generate segmentation masks for each image using SAM.

    Parameters
    ----------
    image_dir : str
        Directory containing input RGB images.
    output_dir : str
        Directory where mask images will be saved.

    Returns
    -------
    List[np.ndarray]
        List of segmentation masks corresponding to each input image.

    Notes
    -----
    Masks will be used for Pipeline A (2D pre-segmentation reconstruction).
    """

    raise NotImplementedError("SAM mask generation not implemented yet.")