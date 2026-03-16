"""
Pipeline A: 2D pre-segmentation reconstruction.

Steps:
1. Generate segmentation masks using SAM
2. Mask images
3. Run COLMAP reconstruction
4. Generate object-level meshes
5. Evaluate reconstruction quality
"""
import yaml
from src.segmentation.sam_masks import generate_masks
from src.reconstruction.colmap_utils import run_sfm, run_dense_reconstruction
from src.evaluation.metrics import evaluate_reconstruction

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():

    config = load_config("configs/pipeline_A.yaml")

    image_dir = config["paths"]["image_dir"]
    mask_dir = config["paths"]["mask_dir"]
    reconstruction_dir = config["paths"]["reconstruction_dir"]

    print("Generating masks...")
    generate_masks(image_dir=image_dir, output_dir=mask_dir)

    print("Running SfM...")
    run_sfm(image_dir=image_dir, output_dir=reconstruction_dir)

    print("Running dense reconstruction...")
    run_dense_reconstruction(workspace_dir=reconstruction_dir)

    if config["evaluation"]["run_metrics"]:
        print("Evaluating reconstruction...")
        evaluate_reconstruction(reconstruction_dir)


if __name__ == "__main__":
    main()