"""
Input / output utilities.

Common file handling utilities used throughout the project.
"""

import os
from typing import List


def ensure_dir(path: str):
    """
    Ensure a directory exists.

    Parameters
    ----------
    path : str
        Directory path.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def list_images(image_dir: str) -> List[str]:
    """
    List image files in a directory.

    Parameters
    ----------
    image_dir : str
        Directory containing images.

    Returns
    -------
    list
        List of image file paths.
    """

    valid_ext = [".jpg", ".jpeg", ".png"]

    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    return sorted(images)


def save_pointcloud(path: str, points):
    """
    Save a point cloud to disk.

    Parameters
    ----------
    path : str
        Output file path.
    points : np.ndarray
        Nx3 point array.
    """

    # TODO implement
    raise NotImplementedError("Point cloud saving not implemented yet.")