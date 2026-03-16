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

## Datasets
- DTU Multi-view stereo dataset
- Synthetic Blender scenes

## Metrics
- Chamfer Distance
- F-score
- Reconstruction completeness

Report: https://www.overleaf.com/6422834493vwhwjqtjhxmv#bbfd5f

Google slides: https://docs.google.com/presentation/d/1lc2SpRrMocpWa_39nG2v4Zsmyz3ZawHgFKSXULrsi1Q/edit?usp=sharing 