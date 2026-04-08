"""
Pipeline A: 2D pre-segmentation reconstruction.

Steps:
1. Generate segmentation masks using SAM
2. Mask images
3. Run COLMAP reconstruction
4. Generate object-level meshes
5. Evaluate reconstruction quality
"""
import os
import yaml
from src.reconstruction.colmap_utils import run_colmap_pipeline
from src.segmentation.sam_masks import run_sam_folder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(BASE_DIR, "configs", "pipeline_A.yaml")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config(config_path)

    # Fix paths relative to project root
    ROOT = BASE_DIR


    def resolve(p):
        return os.path.join(ROOT, p)


    config["data"]["images_path"] = resolve(config["data"]["images_path"])
    config["data"]["masked_images_path"] = resolve(config["data"]["masked_images_path"])

    # 1. Run SAM masking
    print("Running SAM masking...")
    run_sam_folder(config)

    # 2. Switch COLMAP input to masked images
    config["data"]["images_path"] = config["data"]["masked_images_path"]

    # 3. Run COLMAP reconstruction
    print("Running COLMAP on masked images...")
    run_colmap_pipeline(config)