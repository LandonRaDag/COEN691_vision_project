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

## Environment Setup

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
