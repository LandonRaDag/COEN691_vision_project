"""
Pipeline B: 3D instance segmentation after reconstruction.

Steps:
1. Run COLMAP SfM
2. Run dense reconstruction (MVS)
3. Perform 3D instance segmentation using OpenMask3D
4. Extract object-level point clouds
5. Evaluate reconstruction quality
"""

import argparse
import yaml

from src.reconstruction.mvs import run_dense_reconstruction
from src.segmentation.openmask3d_wrapper import segment_pointcloud
from src.utils.io import ensure_dir


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config("configs/pipeline_B.yaml")

    paths = config["paths"]

    image_dir = paths["image_dir"]
    reconstruction_dir = paths["reconstruction_dir"]
    segmentation_dir = paths["segmentation_dir"]

    ensure_dir(reconstruction_dir)
    ensure_dir(segmentation_dir)

    print("Running dense reconstruction...")
    run_dense_reconstruction(
        image_dir=image_dir,
        output_dir=reconstruction_dir,
        config=config["reconstruction"]
    )

    print("Running 3D segmentation...")
    segment_pointcloud(
        pointcloud_dir=reconstruction_dir,
        output_dir=segmentation_dir,
        config=config["segmentation"]
    )

    print("Pipeline B completed.")


if __name__ == "__main__":
    main()