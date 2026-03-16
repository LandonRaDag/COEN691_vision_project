"""
Aggregated metrics to evaluate end results 
"""

import numpy as np

def evaluate_reconstruction(pointcloud_a: np.ndarray, pointcloud_b: np.ndarray):
    """
    Use existing evaluation methods to return a metric (metrics?) 
    for quality

    Parameters
    ----------
    pointcloud_a : np.ndarray
        Nx3 array of predicted points.
    pointcloud_b : np.ndarray
        Mx3 array of ground truth points.

    Returns
    -------
    array? list of all results? combined metric? TBD 
    
    """
    raise NotImplementedError("Chamfer distance not implemented yet.")
    
