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
