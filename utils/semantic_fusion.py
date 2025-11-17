"""
Semantic Fusion Utilities for Road Point Cloud

This module provides functions to fuse multi-view semantic segmentation results
from multiple cameras into a unified 3D road point cloud representation.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
from datasets.nusc import worldpoint2camera


def fuse_multiview_semantics(
    dataset,
    num_classes: int = 65,
    min_distance: float = 1.0,
    use_all_cameras: bool = True,
    max_images: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Fuse multi-view semantic segmentation from all camera images into road point cloud.
    
    This function projects the 3D road point cloud to each camera view, samples the
    semantic labels/features from the segmentation images, and accumulates them
    back to the 3D points. This provides a unified semantic representation that
    incorporates information from all available viewpoints.
    
    Args:
        dataset: NuscDataset instance with loaded road_pointcloud and camera data
        num_classes: Number of semantic classes (default 65 for nuScenes)
        min_distance: Minimum distance threshold for point projection (meters)
        use_all_cameras: Whether to use all 6 cameras or just front camera
        max_images: Maximum number of images to process (None = all), useful for quick testing
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with enriched road point cloud data:
            - "xyz": (N, 3) world coordinates
            - "rgb": (N, 3) original RGB colors
            - "label": (N,) or (N, C) original labels
            - "sem_feat": (N, C) accumulated semantic features from multi-view fusion
            - "view_count": (N,) number of views that observed each point
    """
    # Get road point cloud
    xyz_all = dataset.road_pointcloud["xyz"]  # (N, 3)
    rgb_all = dataset.road_pointcloud["rgb"]  # (N, 3)
    label_all = dataset.road_pointcloud["label"]  # (N,) or (N, C)
    
    N = xyz_all.shape[0]
    
    # Initialize semantic feature accumulator
    sem_feat = np.zeros((N, num_classes), dtype=np.float32)
    view_count = np.zeros(N, dtype=np.int32)  # Track how many views see each point
    
    # Determine which images to process
    total_images = len(dataset)
    if max_images is not None:
        total_images = min(total_images, max_images)
    
    if verbose:
        print(f"Fusing semantics from {total_images} images into {N} road points...")
    
    iterator = range(total_images)
    if verbose:
        iterator = tqdm(iterator, desc="Multi-view semantic fusion")
    
    # Process each camera image
    for img_idx in iterator:
        # Get camera parameters
        cam2world = dataset.camera2world_all[img_idx]  # (4, 4)
        K = dataset.cameras_K_all[img_idx]  # (3, 3)
        cam_idx = dataset.cameras_idx_all[img_idx]
        
        # Optional: skip non-front cameras if use_all_cameras is False
        if not use_all_cameras and cam_idx != 0:  # 0 is typically CAM_FRONT
            continue
        
        # Load segmentation image
        label_rel_path = dataset.label_filenames_all[img_idx]
        
        # Try multiple possible paths for segmentation images
        seg_path_candidates = [
            os.path.join(dataset.image_dir, label_rel_path),
            os.path.join(dataset.base_dir, label_rel_path),
        ]
        
        seg = None
        for seg_path in seg_path_candidates:
            if os.path.exists(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                break
        
        if seg is None:
            # Skip if segmentation not found (could warn in verbose mode)
            continue
        
        H, W = seg.shape
        
        # Project all road points to this camera
        try:
            uv, depths, mask_valid = worldpoint2camera(
                xyz_all,
                WH=(W, H),
                cam2world=cam2world,
                cam_intrinsic=K,
                min_dist=min_distance
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to project points for image {img_idx}: {e}")
            continue
        
        # Sample semantic labels at projected locations
        u = uv[0].astype(np.int32)
        v = uv[1].astype(np.int32)
        
        # Get semantic labels for valid points
        valid_indices = np.where(mask_valid)[0]
        if len(valid_indices) == 0:
            continue
        
        sem_labels = seg[v, u]  # (M_valid,)
        
        # Accumulate semantic features (vectorized one-hot encoding)
        # Filter valid labels
        valid_mask = (sem_labels >= 0) & (sem_labels < num_classes)
        valid_pts = valid_indices[valid_mask]
        valid_labels = sem_labels[valid_mask]
        
        # Use np.add.at for efficient accumulation
        np.add.at(sem_feat, (valid_pts, valid_labels), 1.0)
        np.add.at(view_count, valid_pts, 1)
    
    # Normalize semantic features by view count
    # For points seen by multiple views, average the semantic votes
    mask_seen = view_count > 0
    sem_feat[mask_seen] = sem_feat[mask_seen] / view_count[mask_seen, np.newaxis]
    
    if verbose:
        seen_ratio = mask_seen.sum() / N * 100
        avg_views = view_count[mask_seen].mean() if mask_seen.any() else 0
        print(f"Semantic fusion complete:")
        print(f"  - {mask_seen.sum()}/{N} points ({seen_ratio:.1f}%) visible in at least one view")
        print(f"  - Average views per visible point: {avg_views:.1f}")
    
    return {
        "xyz": xyz_all,
        "rgb": rgb_all,
        "label": label_all,
        "sem_feat": sem_feat,
        "view_count": view_count
    }


def project_points_to_bev(
    xyz: np.ndarray,
    features: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    resolution: float = 0.05,
    aggregation: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points with features to a 2D BEV (Bird's Eye View) grid.
    
    Args:
        xyz: (N, 3) 3D point coordinates
        features: (N, C) features to project (e.g., semantic features, RGB)
        min_xy: (2,) minimum coordinates [x_min, y_min]
        max_xy: (2,) maximum coordinates [x_max, y_max]
        resolution: BEV grid resolution in meters
        aggregation: How to aggregate features in each cell ("mean", "max", or "sum")
        
    Returns:
        bev_grid: (H, W, C) BEV feature grid
        bev_mask: (H, W) binary mask indicating which cells have points
        bev_count: (H, W) number of points in each cell
    """
    # Calculate grid dimensions
    x_range = max_xy[0] - min_xy[0]
    y_range = max_xy[1] - min_xy[1]
    
    W = int(np.ceil(x_range / resolution))
    H = int(np.ceil(y_range / resolution))
    
    C = features.shape[1] if len(features.shape) > 1 else 1
    
    # Initialize BEV grids
    if aggregation == "mean":
        bev_grid = np.zeros((H, W, C), dtype=np.float32)
        bev_count = np.zeros((H, W), dtype=np.int32)
    elif aggregation == "max":
        bev_grid = np.full((H, W, C), -np.inf, dtype=np.float32)
        bev_count = np.zeros((H, W), dtype=np.int32)
    else:  # sum
        bev_grid = np.zeros((H, W, C), dtype=np.float32)
        bev_count = np.zeros((H, W), dtype=np.int32)
    
    # Convert world coordinates to grid indices
    x_idx = ((xyz[:, 0] - min_xy[0]) / resolution).astype(np.int32)
    y_idx = ((xyz[:, 1] - min_xy[1]) / resolution).astype(np.int32)
    
    # Clip to valid range
    x_idx = np.clip(x_idx, 0, W - 1)
    y_idx = np.clip(y_idx, 0, H - 1)
    
    # Aggregate features into BEV grid
    if len(features.shape) == 1:
        features = features[:, np.newaxis]
    
    for i in range(len(xyz)):
        xi, yi = x_idx[i], y_idx[i]
        
        if aggregation == "mean" or aggregation == "sum":
            bev_grid[yi, xi] += features[i]
            bev_count[yi, xi] += 1
        elif aggregation == "max":
            bev_grid[yi, xi] = np.maximum(bev_grid[yi, xi], features[i])
            bev_count[yi, xi] += 1
    
    # Finalize based on aggregation method
    if aggregation == "mean":
        mask = bev_count > 0
        bev_grid[mask] = bev_grid[mask] / bev_count[mask, np.newaxis]
    
    bev_mask = (bev_count > 0).astype(np.uint8)
    
    return bev_grid, bev_mask, bev_count


def create_bev_semantic_image(
    xyz: np.ndarray,
    sem_feat: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    resolution: float = 0.05,
    color_map: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create BEV semantic image from 3D points with semantic features.
    
    Args:
        xyz: (N, 3) 3D point coordinates
        sem_feat: (N, C) semantic probability features
        min_xy: (2,) minimum BEV coordinates
        max_xy: (2,) maximum BEV coordinates
        resolution: BEV resolution in meters
        color_map: Optional (C, 3) RGB color map for visualization
        
    Returns:
        bev_label: (H, W) semantic label image (argmax of sem_feat)
        bev_label_vis: (H, W, 3) colored visualization of labels
        bev_mask: (H, W) valid region mask
    """
    # Project semantic features to BEV
    bev_sem, bev_mask, bev_count = project_points_to_bev(
        xyz, sem_feat, min_xy, max_xy, resolution, aggregation="mean"
    )
    
    # Get most likely class at each location
    bev_label = np.argmax(bev_sem, axis=2).astype(np.uint8)  # (H, W)
    bev_label[bev_mask == 0] = 0  # Set invalid regions to class 0
    
    # Create colored visualization
    if color_map is not None:
        bev_label_vis = color_map[bev_label]
        if len(bev_label_vis.shape) == 4:  # If color_map has extra dimension
            bev_label_vis = bev_label_vis[:, :, 0, :]
    else:
        # Default grayscale visualization
        bev_label_vis = np.stack([bev_label] * 3, axis=-1)
    
    return bev_label, bev_label_vis, bev_mask


def save_bev_results(
    save_dir: str,
    bev_label: np.ndarray,
    bev_label_vis: np.ndarray,
    bev_mask: np.ndarray,
    bev_rgb: Optional[np.ndarray] = None
):
    """
    Save BEV results to disk.
    
    Args:
        save_dir: Directory to save results
        bev_label: (H, W) semantic label image
        bev_label_vis: (H, W, 3) colored visualization
        bev_mask: (H, W) valid mask
        bev_rgb: Optional (H, W, 3) RGB BEV image
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw label
    cv2.imwrite(os.path.join(save_dir, "bev_semantic_label.png"), bev_label)
    
    # Save colored visualization
    bev_label_vis_bgr = cv2.cvtColor(bev_label_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "bev_semantic_vis.png"), bev_label_vis_bgr)
    
    # Save mask
    cv2.imwrite(os.path.join(save_dir, "bev_mask.png"), bev_mask * 255)
    
    # Save RGB if provided
    if bev_rgb is not None:
        bev_rgb_bgr = cv2.cvtColor((bev_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, "bev_rgb.png"), bev_rgb_bgr)
    
    print(f"BEV results saved to {save_dir}")
