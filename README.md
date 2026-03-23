# Object-Centric Multi-View Reconstruction with Segmentation Priors

Project for **COEN 691 – Advanced Robotic Vision**

## Goal
Compare the effect of segmentation priors on object-centric 3D reconstruction.

Two pipelines are evaluated:

1. **2D Pre-Segmentation**
   - SAM segmentation
   - Masked image reconstruction with COLMAP

2. **3D Instance Segmentation**
   - COLMAP reconstruction
   - OpenMask3D instance segmentation
  
<img width="480" height="675" alt="proposal" src="https://github.com/user-attachments/assets/cbf0708f-c65a-4ec8-975b-6f09633d1c15" />


## Datasets
- DTU Multi-view stereo dataset
- Synthetic Blender scenes

## Metrics
- Chamfer Distance
- F-score
- Reconstruction completeness

## Documents
Report: https://www.overleaf.com/6422834493vwhwjqtjhxmv#bbfd5f

Google slides: https://docs.google.com/presentation/d/1lc2SpRrMocpWa_39nG2v4Zsmyz3ZawHgFKSXULrsi1Q/edit?usp=sharing 

## Overall Environment Setup

This project uses COLMAP via CLI. You must configure it correctly for your machine.

### 1. Install COLMAP
Download COLMAP (CUDA version recommended):
https://colmap.github.io/install.html

### 2. Update executable path
Edit:
configs/pipeline_A.yaml

Example:
colmap:
  executable: C:/COLMAP-3.7-windows-cuda/bin/colmap.exe

⚠️ This path MUST match your local installation.

---

### 3. (Optional) CUDA / GPU Acceleration
To enable GPU for dense reconstruction:

- Install NVIDIA CUDA Toolkit (e.g. 11.8)
- Make sure CUDA is available in your system PATH

Check:
nvcc --version

If CUDA is not available, COLMAP will run on CPU (slower but works).

---

### 4. Environment Variables (Windows)

If COLMAP is not recognized, add to PATH:

- COLMAP root folder
- COLMAP /bin
- CUDA /bin (if using GPU)

Example paths:
C:\COLMAP-3.7-windows-cuda\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin

Then restart terminal / IDE.

---

### 5. Verify installation

Run:
colmap -h

You should see COLMAP help output.

---

### 6. Common Issues

- COLMAP not found  
  → Check PATH or YAML executable path  

- Slow reconstruction  
  → Ensure GPU is enabled or reduce image count  

- Empty or failed reconstruction  
  → Ensure images have overlap (multi-view consistency)

- Database errors  
  → Ensure data/colmap_workspace/ exists

## Pipeline B Environment Setup

This project uses OpenMask3D as an external dependency. OpenMask3D is installed in the external folder, some required paths must be configured for your machine before running Pipeline B.

### 1. Install OpenMask3D dependencies

OpenMask3D requires its own dependencies. Install them in the project environment.

Recommended setup from the OpenMask3D repository:

```cd external/OpenMask3D
bash install_requirements.sh
pip install -e .
```

### 2. Download required checkpoints 

Create the required checkpoint files in the OpenMask3D resources folder:

``external/OpenMask3D/resources/``

Required files:

- [mask module checkpoint](https://drive.google.com/file/d/1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y/view)
- [SAM checkpoint](https://drive.google.com/file/d/1WHi0hBi0iqMZfk8l3rDXLrW4lEEgHm_y/view)

Example expected paths:

```external/OpenMask3D/resources/scannet200_model.ckpt
external/OpenMask3D/resources/sam_vit_h_4b8939.pth
```

These filenames and locations must match the values in the YAML config.

### 3. Update YAML configuration
Edit: ``configs/pipeline_B.yaml``

Update the following fields under ``openmask3d:`` to match your machine and scene:

Example:

```
openmask3d:
  scene_dir: data/openmask3d/scene_example
  scene_ply_name: scene_example.ply
  intrinsic_resolution: "[968,1296]"
  img_extension: .jpg
  depth_extension: .png
  depth_scale: 1000
  mask_module_ckpt_path: external/OpenMask3D/resources/scannet200_model.ckpt
  sam_ckpt_path: external/OpenMask3D/resources/sam_vit_h_4b8939.pth
  output_directory: results/openmask3d
  experiment_name: experiment
  save_visualizations: false
  save_crops: false
  optimize_gpu_usage: false
  ```

Fields to check carefully
- scene_dir; Path to the scene folder
- scene_ply_name; Name of the point cloud file inside scene_dir
- intrinsic_resolution; Resolution used when computing camera intrinsics
- img_extension / depth_extension; Must match your actual file types
- depth_scale; Must match the depth format of your dataset
- mask_module_ckpt_path; Path to the mask module checkpoint
- sam_ckpt_path; Path to the SAM checkpoint
- output_directory; Where Pipeline B results will be saved

These paths must match your local setup.

### 4. Verify installation

Before running Pipeline B, verify that:

- OpenMask3D repo exists at external/OpenMask3D
- checkpoints exist
- scene directory exists
- the expected subfolders are present
- the Python environment has all OpenMask3D dependencies installed

You may also test from inside the OpenMask3D repo: