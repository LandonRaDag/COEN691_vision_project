
# Object-Centric Multi-View Reconstruction with Segmentation Priors

Project for **COEN 691 – Advanced Robotic Vision**

---

## Goal

Compare the effect of segmentation priors on object-centric 3D reconstruction.

Two pipelines are evaluated:

1. **2D Pre-Segmentation (Pipeline A)**

   * Segment Anything Model (SAM)
   * Masked image reconstruction with COLMAP

2. **3D Instance Segmentation (Pipeline B)**

   * COLMAP reconstruction
   * OpenMask3D instance segmentation

The objective is to evaluate how segmentation placement (before vs after reconstruction) affects geometric quality and robustness.

<img width="480" height="675" alt="proposal" src="https://github.com/user-attachments/assets/cbf0708f-c65a-4ec8-975b-6f09633d1c15" />

---

## Dataset

* **CO3D (Common Objects in 3D)**
  Provides:

  * Multi-view RGB images
  * Ground-truth masks
  * Depth maps
  * Camera parameters

Subset used:

* car / pizza / plant / teddy / wineglass (selected scenes)

---

## Evaluation (Final Approach)

Initial evaluation using point cloud metrics (Chamfer, F-score) was abandoned due to instability from alignment issues.

Final evaluation uses **reprojection-based metrics**:

* **Mask IoU**

  * Compare projected reconstruction vs ground-truth masks

* **Depth MAE (scale-normalized)**

  * Compare projected depth vs CO3D depth maps
  * Scale ambiguity resolved via per-frame scalar fitting

### Key idea

Instead of aligning point clouds globally, we:

1. Project reconstructed 3D points into each image
2. Compare directly in image space

This avoids:

* ICP instability
* scale mismatch issues
* coordinate frame ambiguity

---

## Results Summary

| Pipeline            | Mean Mask IoU | Mean Depth MAE (scaled) |
| ------------------- | ------------- | ----------------------- |
| A (2D Segmentation) | 0.7227        | 5120.116                |
| B (3D Segmentation) | 0.4724        | 7519.821                |

Observations:

* Pipeline A produces cleaner object-focused reconstructions
* Pipeline B suffers from noisy segmentation and background leakage
* Reprojection confirms good global structure but local inconsistencies

---

## Visual Outputs

Evaluation generates:

* RGB + projection overlay
* Predicted vs GT masks
* Depth comparison (pred / GT)
* Depth error heatmaps
* Summary panels per frame

Example outputs are saved in:

```
results/pipeline A/<scene>/metrics_eval/
results/pipeline B/<scene>/metrics_eval/
```

---

## Environment Setup

This project uses COLMAP via CLI. You must configure it correctly for your machine.

---

### 1. Install COLMAP

Download COLMAP (CUDA version recommended):
[https://colmap.github.io/install.html](https://colmap.github.io/install.html)

---

### 2. Update executable path

Edit:

```
configs/pipeline_A.yaml
```

Example:

```yaml
colmap:
  executable: C:/COLMAP-3.7-windows-cuda/bin/colmap.exe
```

⚠️ This path MUST match your local installation.

---

### 3. (Optional) CUDA / GPU Acceleration

To enable GPU for dense reconstruction:

* Install NVIDIA CUDA Toolkit (e.g. 11.8)
* Ensure CUDA is in system PATH

Check:

```
nvcc --version
```

---

### 4. Environment Variables (Windows)

If COLMAP is not recognized, add to PATH:

* COLMAP root folder
* COLMAP `/bin`
* CUDA `/bin` (if using GPU)

Example:

```
C:\COLMAP-3.7-windows-cuda\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

Restart terminal after updating.

---

### 5. Verify installation

Run:

```
colmap -h
```

You should see help output.

---

## Running Pipelines

### Pipeline A (2D Segmentation + Reconstruction)

```
python scripts/run_pipeline_A.py
```

### Pipeline B (3D Segmentation)

* Run COLMAP reconstruction first
* Then run OpenMask3D (Linux/WSL recommended)

---

## Running Evaluation

```
python scripts/run_metrics.py
```

This will:

* Project 3D points into image space
* Compute IoU and depth MAE
* Generate visual outputs
* Save metrics summary

---

## Common Issues


- COLMAP not found  
  → Check PATH or YAML executable path  

- Slow reconstruction  
  → Ensure GPU is enabled or reduce image count  

- Empty or failed reconstruction  
  → Ensure images have overlap (multi-view consistency)

- Database errors  
  → Ensure data/colmap_workspace/ exists
