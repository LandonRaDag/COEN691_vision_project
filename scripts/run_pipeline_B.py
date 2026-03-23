"""
Pipeline B: 3D instance segmentation after reconstruction.

Steps:
1. Run COLMAP SfM
2. Run dense reconstruction (MVS)
3. Perform 3D instance segmentation using OpenMask3D
4. Extract object-level point clouds
5. Evaluate reconstruction quality
"""
import os
import yaml
from src.reconstruction.colmap_utils import run_colmap_pipeline
from src.segmentation.openmask3d_wrapper import run_openmask3d_pipeline


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(BASE_DIR, "configs", "pipeline_A.yaml")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config(config_path)
    run_colmap_pipeline(config)
    
    if config.get("openmask3d", {}).get("enabled", False):
        run_openmask3d_pipeline(config)