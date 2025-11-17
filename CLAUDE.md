# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoGs (Road Surface Reconstruction with Meshgrid Gaussian) is a PyTorch-based computer vision research project for large-scale road surface reconstruction using 3D Gaussian splatting. It's designed for autonomous driving datasets (nuScenes and KITTI) and implements adaptive grid optimization with KD-tree spatial partitioning.

## Common Development Commands

### Environment Setup
```bash
# Activate conda environment
conda activate rogs

# Install dependencies (if needed)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install addict PyYAML tqdm scipy pytz plyfile opencv-python pyrotation pyquaternion nuscenes-devkit
```

### Training
```bash
# Train on nuScenes mini dataset
python train.py --config configs/local_nusc_mini.yaml

# Train on full nuScenes dataset
python train.py --config configs/local_nusc.yaml

# Train on KITTI dataset
python train.py --config configs/local_kitti.yaml
```

### Evaluation and Analysis
```bash
# Run parameter tuning for KD-tree optimization
python run_parameter_tuning.py --percentiles 1 2 3 5 10

# Compare grid methods (KD-tree vs fixed grid)
python run_grid_comparison.py

# Visualize grid comparison results
python visualize_grid_comparison.py

# Interactive KD-tree parameter adjustment
bash run_kdtree_tools.sh
```

### Data Preprocessing
```bash
# Generate ground truth for nuScenes
python -m preprocess.process --nusc_root /dataset/nuScenes/v1.0-mini --seg_root /dataset/nuScenes/nuScenes_clip --save_root /dataset/nuScenes/ -v mini --scene_names scene-0655
```

## Architecture and Key Components

### Core Model Architecture
- **Gaussian Splatting Pipeline**: Uses custom CUDA rasterization with orthographic camera support for BEV rendering
- **Adaptive Grid System**: KD-tree based spatial partitioning with integral image optimization (30x+ speed improvement)
- **Multi-model Support**: Original `road.py`, adaptive `adaptive_road.py`, and optimized `adaptive_road_v2.py`

### Key Model Files
- `models/road.py`: Base road reconstruction model with hexagonal/rectangular mesh generation
- `models/adaptive_road_v2.py`: Latest version with KD-tree V2 and integral image optimization
- `models/kdtree_v2.py`: Optimized KD-tree implementation with semantic fusion
- `models/gaussian_model.py`: Core 3D Gaussian splatting models

### Dataset Integration
- `datasets/nusc.py`: nuScenes dataset loader with multi-camera support (6 cameras)
- `datasets/kitti.py`: KITTI odometry dataset loader
- `datasets/base.py`: Base dataset class with common functionality

### Configuration System
- YAML-based configuration in `configs/` directory
- Key parameters: `bev_resolution`, `cut_range`, `sh_degree`, various learning rates
- Dataset-specific configs for nuScenes mini/full and KITTI

### Performance Optimizations
- **KD-tree V2**: Integral image-based adaptive grid with configurable `auto_threshold_percentile`
- **Block-based Rendering**: Spatial partitioning for memory efficiency
- **CUDA Custom Extensions**: diff-gaussian-rasterization with orthographic camera support
- **Semantic Channel Optimization**: Separate libraries for RGB and semantic rendering

## Important Technical Details

### Language and Comments
- Mixed English/Chinese codebase (comments often in Chinese)
- Key technical terms may be in Chinese, especially in recent KD-tree implementations

### Dependencies
- PyTorch 1.13.1 with CUDA 11.7
- PyTorch3D 0.7.8 for 3D operations
- Custom CUDA extensions required (must be compiled separately)
- No traditional package manager (no requirements.txt)

### Dataset Requirements
- **nuScenes**: Official dataset + semantic segmentation results + generated ground truth
- **KITTI**: Odometry dataset + semantic segmentation + pose data
- Semantic segmentation results should be placed in `image_dir` configuration

### Recent Development Focus
- KD-tree optimization for adaptive grid systems
- Performance improvements (30x+ speedup claimed)
- Integral image-based spatial partitioning
- Parameter tuning for minimum area resolution (0.0025m² target)

## Common Issues and Solutions

### CUDA Extension Compilation
Custom CUDA extensions must be compiled separately:
```bash
git clone --recursive https://github.com/fzhiheng/diff-gs-depth-alpha.git
cd diff-gs-depth-alpha
python setup.py install
```

### Semantic Rendering
Two separate libraries needed:
- `diff_gaussian_rasterization`: For RGB rendering
- `diff_gs_label`/`diff_gs_label2`: For semantic segmentation (different channel counts)

### Memory Management
- Use block-based rendering for large scenes
- Adjust `cut_range` parameter to control memory usage
- KD-tree V2 significantly reduces memory requirements

## Testing and Validation

No formal testing framework exists. Validation is done through:
- Visual inspection of reconstruction quality
- Quantitative metrics (PSNR, depth accuracy)
- Grid comparison experiments
- Parameter tuning scripts with timing benchmarks