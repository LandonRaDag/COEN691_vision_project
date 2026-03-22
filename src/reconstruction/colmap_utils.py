import subprocess
import os

# Set up environment variables for COLMAP and CUDA
ENV = os.environ.copy()
ENV["PATH"] += r";C:\COLMAP-3.7-windows-cuda"
ENV["PATH"] += r";C:\COLMAP-3.7-windows-cuda\lib"
ENV["PATH"] += r";C:\COLMAP-3.7-windows-cuda\bin"
ENV["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"


def run_colmap_pipeline(config):

    fused_path = config["colmap"]["fused_path"]
    if os.path.exists(fused_path):
        print("COLMAP output already exists. Skipping reconstruction.")
        return

    print("Running COLMAP pipeline...")

    COLMAP = config["colmap"]["executable"]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    db_path = os.path.join(BASE_DIR, config["colmap"]["database_path"])
    image_path = os.path.join(BASE_DIR, config["data"]["images_path"])
    sparse_path = os.path.join(BASE_DIR, config["colmap"]["sparse_path"])
    dense_path = os.path.join(BASE_DIR, config["colmap"]["dense_path"])
    fused_path = os.path.join(BASE_DIR, config["colmap"]["fused_path"])

    run_feature_extraction(COLMAP, db_path, image_path)
    run_feature_matching(COLMAP, db_path)
    run_mapper(COLMAP, db_path, image_path, sparse_path)

    sparse_model_path = os.path.join(sparse_path, "0")

    run_image_undistorter(COLMAP, image_path, sparse_model_path, dense_path)
    run_patch_match(COLMAP, dense_path)
    run_stereo_fusion(COLMAP,dense_path, fused_path)

def run_feature_extraction(colmap, db_path, image_path):
    subprocess.run([
        colmap, "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_path
    ], check=True, env=ENV)

def run_feature_matching(colmap, db_path):
    subprocess.run([
        colmap, "exhaustive_matcher",
        "--database_path", db_path
    ], check=True, env=ENV)

def run_mapper(colmap, db_path, image_path, sparse_path):
    os.makedirs(sparse_path, exist_ok=True)
    subprocess.run([
        colmap, "mapper",
        "--database_path", db_path,
        "--image_path", image_path,
        "--output_path", sparse_path
    ], check=True, env=ENV)

def run_image_undistorter(colmap, image_path, sparse_model_path, dense_path):
    subprocess.run([
        colmap, "image_undistorter",
        "--image_path", image_path,
        "--input_path", sparse_model_path,
        "--output_path", dense_path,
        "--output_type", "COLMAP"
    ], check=True, env=ENV)

def run_patch_match(colmap, dense_path):
    subprocess.run([
        colmap, "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.max_image_size", "500",
        "--PatchMatchStereo.window_radius", "3"
    ], check=True, env=ENV)

def run_stereo_fusion(colmap, dense_path, output_path):
    subprocess.run([
        colmap, "stereo_fusion",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", output_path
    ], check=True, env=ENV)