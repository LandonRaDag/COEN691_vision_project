"""
OpenMask3D segmentation wrapper.

This module interfaces with OpenMask3D to perform 3D instance
segmentation on reconstructed point clouds.
"""

from typing import Dict


def segment_pointcloud(
    pointcloud_dir: str,
    output_dir: str,
    config: Dict
):
    """
    Perform 3D instance segmentation on a point cloud.

    Parameters
    ----------
    pointcloud_dir : str
        Directory containing reconstructed point cloud data.

    output_dir : str
        Directory where segmented objects will be stored.

    config : dict
        Segmentation parameters from YAML config.

    Notes
    -----
    Expected workflow:

    1. Load reconstructed point cloud
    2. Run OpenMask3D inference
    3. Extract instance clusters
    4. Save each object as separate point cloud
    """

    print("Running OpenMask3D segmentation...")

    # TODO
    # - load point cloud
    # - run OpenMask3D model
    # - extract instance clusters

    raise NotImplementedError("OpenMask3D wrapper not implemented yet.")