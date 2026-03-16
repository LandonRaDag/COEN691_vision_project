"""
Chamfer distance computation for evaluating reconstruction accuracy.
"""

import numpy as np


def chamfer_distance(pointcloud_a: np.ndarray, pointcloud_b: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point clouds.

    Parameters
    ----------
    pointcloud_a : np.ndarray
        Nx3 array of predicted points.
    pointcloud_b : np.ndarray
        Mx3 array of ground truth points.

    Returns
    -------
    float
        Chamfer distance metric.
    """

    raise NotImplementedError("Chamfer distance not implemented yet.")