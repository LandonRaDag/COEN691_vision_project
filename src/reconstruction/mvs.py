"""
Multi-view stereo reconstruction utilities.

This module handles dense reconstruction of scenes using COLMAP
"""

import subprocess
from typing import Dict


def run_dense_reconstruction(
    image_dir: str,
    output_dir: str,
    config: Dict
):
    """
    Run dense multi-view stereo reconstruction.

    Parameters
    ----------
    image_dir : str
        Directory containing input images.

    output_dir : str
        Directory where reconstruction results will be stored.

    config : dict
        Reconstruction parameters from YAML config.

    Notes
    -----
    Probbaly just run COLMAP 
        - image undistortion
        - patch match stereo
        - stereo fusion
    """

    print("Starting dense reconstruction...")

    # TODO: implement full COLMAP pipeline
    # Example steps:
    #
    # 1. image_undistorter
    # 2. patch_match_stereo
    # 3. stereo_fusion

    raise NotImplementedError("Dense reconstruction pipeline not implemented yet.")