import subprocess
import os
from datetime import datetime

# Optional environment setup, matching the style of colmap_utils.py
ENV = os.environ.copy()
ENV["OMP_NUM_THREADS"] = "3"


def run_openmask3d_pipeline(config):
    print("Running OpenMask3D pipeline...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    OPENMASK3D_DIR = os.path.join(BASE_DIR, "external", "OpenMask3D")

    SCENE_DIR = os.path.join(BASE_DIR, config["openmask3d"]["scene_dir"])
    SCENE_POSE_DIR = os.path.join(SCENE_DIR, "pose")
    SCENE_INTRINSIC_PATH = os.path.join(SCENE_DIR, "intrinsic", "intrinsic_color.txt")
    SCENE_PLY_PATH = os.path.join(SCENE_DIR, config["openmask3d"]["scene_ply_name"])
    SCENE_COLOR_IMG_DIR = os.path.join(SCENE_DIR, "color")
    SCENE_DEPTH_IMG_DIR = os.path.join(SCENE_DIR, "depth")

    SCENE_INTRINSIC_RESOLUTION = config["openmask3d"]["intrinsic_resolution"]
    IMG_EXTENSION = config["openmask3d"]["img_extension"]
    DEPTH_EXTENSION = config["openmask3d"]["depth_extension"]
    DEPTH_SCALE = config["openmask3d"]["depth_scale"]

    MASK_MODULE_CKPT_PATH = os.path.join(BASE_DIR, config["openmask3d"]["mask_module_ckpt_path"])
    SAM_CKPT_PATH = os.path.join(BASE_DIR, config["openmask3d"]["sam_ckpt_path"])

    OUTPUT_DIRECTORY = os.path.join(BASE_DIR, config["openmask3d"]["output_directory"])
    EXPERIMENT_NAME = config["openmask3d"]["experiment_name"]
    SAVE_VISUALIZATIONS = str(config["openmask3d"]["save_visualizations"]).lower()
    SAVE_CROPS = str(config["openmask3d"]["save_crops"]).lower()
    OPTIMIZE_GPU_USAGE = str(config["openmask3d"]["optimize_gpu_usage"]).lower()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    OUTPUT_FOLDER_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, f"{timestamp}-{EXPERIMENT_NAME}")

    if os.path.exists(OUTPUT_FOLDER_DIRECTORY):
        print("OpenMask3D output already exists. Skipping.")
        return

    # Basic checks
    required_paths = [
        OPENMASK3D_DIR,
        SCENE_DIR,
        SCENE_POSE_DIR,
        SCENE_INTRINSIC_PATH,
        SCENE_PLY_PATH,
        SCENE_COLOR_IMG_DIR,
        SCENE_DEPTH_IMG_DIR,
        MASK_MODULE_CKPT_PATH,
        SAM_CKPT_PATH,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER_DIRECTORY, exist_ok=True)

    # 1. Compute class-agnostic masks
    print("[INFO] Extracting class-agnostic masks...")
    subprocess.run(
        [
            "python",
            "class_agnostic_mask_computation/get_masks_single_scene.py",
            f"general.experiment_name={EXPERIMENT_NAME}",
            f"general.checkpoint={MASK_MODULE_CKPT_PATH}",
            "general.train_mode=false",
            "data.test_mode=test",
            "model.num_queries=120",
            "general.use_dbscan=true",
            "general.dbscan_eps=0.95",
            f"general.save_visualizations={SAVE_VISUALIZATIONS}",
            f"general.scene_path={SCENE_PLY_PATH}",
            f"general.mask_save_dir={OUTPUT_FOLDER_DIRECTORY}",
            f"hydra.run.dir={os.path.join(OUTPUT_FOLDER_DIRECTORY, 'hydra_outputs', 'class_agnostic_mask_computation')}",
        ],
        cwd=OPENMASK3D_DIR,
        env=ENV,
        check=True,
    )

    print("[INFO] Mask computation done!")

    # Build mask path
    ply_base = os.path.basename(SCENE_PLY_PATH)
    mask_file_name = ply_base.replace(".ply", "_masks.pt")
    SCENE_MASK_PATH = os.path.join(OUTPUT_FOLDER_DIRECTORY, mask_file_name)

    if not os.path.exists(SCENE_MASK_PATH):
        raise FileNotFoundError(f"Expected mask file was not created: {SCENE_MASK_PATH}")

    print(f"[INFO] Masks saved to {SCENE_MASK_PATH}")

    # 2. Compute mask features
    print("[INFO] Computing mask features...")
    subprocess.run(
        [
            "python",
            "compute_features_single_scene.py",
            f"data.masks.masks_path={SCENE_MASK_PATH}",
            f"data.camera.poses_path={SCENE_POSE_DIR}",
            f"data.camera.intrinsic_path={SCENE_INTRINSIC_PATH}",
            f"data.camera.intrinsic_resolution={SCENE_INTRINSIC_RESOLUTION}",
            f"data.depths.depths_path={SCENE_DEPTH_IMG_DIR}",
            f"data.depths.depth_scale={DEPTH_SCALE}",
            f"data.depths.depths_ext={DEPTH_EXTENSION}",
            f"data.images.images_path={SCENE_COLOR_IMG_DIR}",
            f"data.images.images_ext={IMG_EXTENSION}",
            f"data.point_cloud_path={SCENE_PLY_PATH}",
            f"output.output_directory={OUTPUT_FOLDER_DIRECTORY}",
            f"output.save_crops={SAVE_CROPS}",
            f"hydra.run.dir={os.path.join(OUTPUT_FOLDER_DIRECTORY, 'hydra_outputs', 'mask_features_computation')}",
            f"external.sam_checkpoint={SAM_CKPT_PATH}",
            f"gpu.optimize_gpu_usage={OPTIMIZE_GPU_USAGE}",
        ],
        cwd=OPENMASK3D_DIR,
        env=ENV,
        check=True,
    )

    print("[INFO] OpenMask3D pipeline complete.")
    print(f"[INFO] Outputs saved to: {OUTPUT_FOLDER_DIRECTORY}")