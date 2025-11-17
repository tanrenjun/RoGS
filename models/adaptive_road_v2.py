"""
Adaptive Road class using KD-tree V2 (integral map optimization)
Uses the optimized KD-tree implementation for efficient adaptive grid generation
"""

import random
import torch
import numpy as np

try:
    import mayavi.mlab as mlab
    from utils.vis import plot_gaussion_3d
except ImportError:
    pass
import cv2
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.transforms import matrix_to_quaternion
from diff_gaussian_rasterization.scene.cameras import OrthographicCamera

from models.kdtree_v2 import AdaptiveKDTreeV2


class AdaptiveRoadV2:
    """
    Adaptive Road class using optimized KD-tree V2.
    Generates adaptive grid vertices based on geometric and semantic features.
    """
    
    def __init__(self, config, dataset, device='cuda:0', vis=False):
        self.device = device
        self.resolution = config["bev_resolution"]  # Base resolution (e.g., 0.05m)
        self.cut_range = config["cut_range"]
        
        # KD-tree V2 configuration
        self.kdtree_config = {
            'bev_resolution': config.get("kdtree_bev_resolution", 0.05),
            'min_area': config.get("kdtree_min_area", 0.0025),  # 0.05m * 0.05m
            'max_depth': config.get("kdtree_max_depth", 18),
            'geo_weight': config.get("kdtree_geo_weight", 0.7),
            'sem_weight': config.get("kdtree_sem_weight", 0.3),
            'auto_threshold_percentile': config.get("kdtree_percentile", 5.0),
            'use_trajectory_crop': config.get("kdtree_use_trajectory_crop", True),
        }
        
        print("\n" + "="*80)
        print("初始化自适应Road（KD-tree V2）")
        print("="*80)
        
        # 获取所有轨迹点和类别信息
        all_poses, num_classes = dataset.chassis2world_unique, dataset.num_class
        all_pose_xyz = all_poses[:, :3, 3]  # 提取xyz坐标
        self.ref_pose = torch.from_numpy(dataset.ref_pose).float().to(device)
        
        # Calculate bounds with cut range
        min_coords = np.min(all_pose_xyz, axis=0) - self.cut_range
        max_coords = np.max(all_pose_xyz, axis=0) + self.cut_range
        self.min_z = np.min(all_pose_xyz[:, -1])
        self.max_z = np.max(all_pose_xyz[:, -1])
        self.min_xy = min_coords[:2]
        self.max_xy = max_coords[:2]

        box = max_coords - min_coords
        self.bev_x_length = box[0]
        self.bev_y_length = box[1]
        
        print(f"BEV范围: {self.bev_x_length:.2f}m x {self.bev_y_length:.2f}m")
        print(f"高度范围: [{self.min_z:.2f}m, {self.max_z:.2f}m]")

        # Create adaptive mesh using KD-tree V2
        traj_point = torch.from_numpy(all_pose_xyz).float().to(device)
        traj_rotation = torch.from_numpy(all_poses[:, :3, :3]).float().to(device)
        
        vertices, four_indices = self._create_adaptive_mesh_v2(
            min_coords, max_coords, 
            all_pose_xyz,
            dataset,
            traj_point,
            traj_rotation,
            num_classes,
            vis
        )
        
        # Store the generated vertices and related data
        self.vertices = vertices  # (M, 3)
        self.four_indices = four_indices.to(device)
        # self.rotation is already set in _create_adaptive_mesh_v2
        self.rgb = torch.zeros_like(self.vertices, device=device)
        self.label = torch.zeros((self.vertices.shape[0], num_classes), dtype=torch.float32, device=device)
        
        # Create BEV camera
        self._create_bev_camera(min_coords, max_coords)
        
        print("="*80)
        print(f"自适应Road初始化完成 - 顶点数: {vertices.shape[0]:,}")
        print("="*80 + "\n")

    def _create_adaptive_mesh_v2(self, min_coords, max_coords, all_pose_xyz, dataset, 
                                  traj_point, traj_rotation, num_classes, vis=False):
        """
        Create adaptive mesh using KD-tree V2 (integral map optimization)
        
        Args:
            min_coords: Minimum coordinates [x_min, y_min, z_min] (from trajectory + cut_range)
            max_coords: Maximum coordinates [x_max, y_max, z_max] (from trajectory + cut_range)
            all_pose_xyz: All pose positions [N, 3] (numpy)
            dataset: Dataset object with road_pointcloud
            traj_point: Trajectory points [N, 3] (torch tensor)
            traj_rotation: Trajectory rotations [N, 3, 3] (torch tensor)
            num_classes: Number of semantic classes
            vis: Whether to visualize
            
        Returns:
            Tuple of (vertices, four_indices)
        """
        print("\n[1/4] 构建KD-tree自适应网格...")
        
        # Get road point cloud from dataset
        road_pointcloud = dataset.road_pointcloud
        if road_pointcloud is None:
            print("警告: 没有路面点云数据，使用固定网格")
            return self._fallback_fixed_grid(min_coords, max_coords, 
                                            traj_point, traj_rotation, vis)
        
        # Extract points and features
        road_xyz = road_pointcloud["xyz"]  # (N, 3)
        road_rgb = road_pointcloud.get("rgb", np.zeros_like(road_xyz))  # (N, 3)
        road_label = road_pointcloud.get("label", np.zeros((road_xyz.shape[0], 1)))  # (N, 1)
        
        # Create semantic feature (use label directly, add small epsilon to avoid zero division)
        # If no labels, use zeros
        sem_feat = torch.from_numpy(road_label + 1e-6).float()  # Add epsilon to avoid nan
        
        # Project to BEV (x, y coordinates)
        bev_points = road_xyz[:, :2]  # (N, 2)
        bev_heights = road_xyz[:, 2:3]  # (N, 1) - height as geometric feature
        
        print(f"  路面点数: {road_xyz.shape[0]:,}")
        print(f"  使用轨迹定义的BEV范围: [{self.min_xy[0]:.2f}, {self.max_xy[0]:.2f}] x "
              f"[{self.min_xy[1]:.2f}, {self.max_xy[1]:.2f}]")
        
        # Initialize KD-tree V2
        kdtree = AdaptiveKDTreeV2(
            bev_resolution=self.kdtree_config['bev_resolution'],
            min_area=self.kdtree_config['min_area'],
            max_depth=self.kdtree_config['max_depth'],
            geo_weight=self.kdtree_config['geo_weight'],
            sem_weight=self.kdtree_config['sem_weight'],
            auto_threshold_percentile=self.kdtree_config['auto_threshold_percentile']
        )
        
        # Build tree with trajectory cropping
        print("\n[2/4] 构建KD-tree...")
        if self.kdtree_config['use_trajectory_crop']:
            from utils.trajectory_utils import create_trajectory_mask
            
            # Use trajectory-based BEV range (already computed in __init__)
            # This matches the original Road class behavior
            traj_mask, mask_shape = create_trajectory_mask(
                all_pose_xyz[:, :2],  # trajectory positions (x, y)
                self.min_xy,  # Use trajectory-based min (not road point min)
                self.max_xy,  # Use trajectory-based max (not road point max)
                self.kdtree_config['bev_resolution'],
                cut_range=self.cut_range  # Use same cut_range as Road class
            )
            print(f"  轨迹掩码覆盖率: {traj_mask.sum() / traj_mask.size * 100:.2f}%")
        else:
            traj_mask = None
            mask_shape = None
        
        # Build the tree
        # Note: Pass predefined BEV range (trajectory-based) to avoid re-cropping
        kdtree.build(
            boundary_points=bev_points,
            sem_feat=sem_feat,
            trajectory_mask=traj_mask,
            cut_range=self.cut_range,
            bev_min=self.min_xy,  # Use trajectory-based range (already includes cut_range)
            bev_max=self.max_xy   # Use trajectory-based range (already includes cut_range)
        )
        
        # Get grid cells from tree (use leaf nodes)
        print("\n[3/4] 提取网格顶点...")
        grid_cells = kdtree.leaves  # Get all leaf nodes
        
        print(f"  网格单元数: {len(grid_cells):,}")
        
        # Convert grid cells to vertices (use cell centers)
        vertices_list = []
        cell_sizes = []
        
        for cell in grid_cells:
            # Use node's min_coords and max_coords
            x_min, y_min = cell.min_coords
            x_max, y_max = cell.max_coords
            # Use cell center as vertex position
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            vertices_list.append([center_x, center_y, 0.0])  # z will be set later
            
            # Store cell size for later use (optional)
            cell_sizes.append((x_max - x_min, y_max - y_min))
        
        vertices = np.array(vertices_list, dtype=np.float32)  # (M, 3)
        print(f"  生成顶点数: {vertices.shape[0]:,}")
        
        # Convert to torch tensor
        vertices = torch.from_numpy(vertices).to(self.device)
        
        # Initialize z coordinates and rotations from nearest trajectory points
        print("\n[4/4] 初始化高度和旋转...")
        nearst_result = knn_points(vertices[None, :, :2], traj_point[None, :, :2], K=1)
        near_points = knn_gather(traj_point[None], nearst_result.idx)[0]  # (M, 1, 3)
        
        nearest_idx = nearst_result.idx[0, :, 0]
        init_z = near_points[:, 0, 2]  # (M,)
        vertices[:, 2] = init_z
        
        rotation = traj_rotation[nearest_idx]  # (M, 3, 3)
        self.rotation = matrix_to_quaternion(rotation)  # (M, 4)
        
        # Create neighbor indices (for smooth loss)
        # For now, use a simple approach: find 4 nearest neighbors
        knn_result = knn_points(vertices[None, :, :2], vertices[None, :, :2], K=5)  # K=5 (self + 4 neighbors)
        four_indices = knn_result.idx[0, :, 1:]  # (M, 4) - exclude self (first one)
        
        print(f"  顶点高度范围: [{vertices[:, 2].min().item():.2f}m, {vertices[:, 2].max().item():.2f}m]")
        
        # Visualization
        if vis:
            self._visualize_adaptive_mesh(vertices, rotation, cell_sizes, min_coords)
        
        return vertices, four_indices
    
    def _fallback_fixed_grid(self, min_coords, max_coords, traj_point, traj_rotation, vis):
        """
        Fallback to fixed grid if road pointcloud is not available
        """
        print("使用固定网格作为后备方案")
        
        box = max_coords - min_coords
        num_vertices_x = int(box[0] / self.resolution) + 1
        num_vertices_y = int(box[1] / self.resolution) + 1
        
        vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
        vertices[:, :, 0] = torch.linspace(min_coords[0], max_coords[0], num_vertices_x).unsqueeze(1)
        vertices[:, :, 1] = torch.linspace(min_coords[1], max_coords[1], num_vertices_y).unsqueeze(0)
        vertices = vertices.reshape(-1, 3).to(self.device)
        
        # Initialize z and rotation
        nearst_result = knn_points(vertices[None, :, :2], traj_point[None, :, :2], K=1)
        near_points = knn_gather(traj_point[None], nearst_result.idx)[0]
        
        nearest_idx = nearst_result.idx[0, :, 0]
        init_z = near_points[:, 0, 2]
        vertices[:, 2] = init_z
        
        rotation = traj_rotation[nearest_idx]
        self.rotation = matrix_to_quaternion(rotation)
        
        # Create neighbor indices
        knn_result = knn_points(vertices[None, :, :2], vertices[None, :, :2], K=5)
        four_indices = knn_result.idx[0, :, 1:]
        
        return vertices, four_indices

    def _create_bev_camera(self, min_coords, max_coords):
        """
        Create orthographic BEV camera for rendering
        """
        SLACK_Z = 1
        box = max_coords - min_coords
        mid_xy = (min_coords[:2] + max_coords[:2]) / 2
        
        bevcam2world = np.array([
            [1, 0, 0, mid_xy[0]],
            [0, -1, 0, mid_xy[1]],
            [0, 0, -1, self.max_z + SLACK_Z],
            [0, 0, 0, 1]
        ])
        bevcam2world = torch.from_numpy(bevcam2world).float()

        render_resolution = self.resolution
        width = int(box[0] / render_resolution) + 1
        height = int(box[1] / render_resolution) + 1
        
        self.bev_camera = OrthographicCamera(
            R=bevcam2world[:3, :3],
            T=bevcam2world[:3, 3],
            W=width,
            H=height,
            znear=0,
            zfar=bevcam2world[2, 3] - self.min_z + SLACK_Z,
            top=-self.bev_y_length * 0.5,
            bottom=self.bev_y_length * 0.5,
            right=self.bev_x_length * 0.5,
            left=-self.bev_x_length * 0.5,
            device=self.device
        )
    
    def _visualize_adaptive_mesh(self, vertices, rotation, cell_sizes, min_coords):
        """
        Visualize the adaptive mesh using Mayavi
        """
        try:
            points = vertices.cpu().numpy()
            sample_idx = random.sample(range(min(points.shape[0], 150)), min(points.shape[0], 150))
            
            sample_points = points[sample_idx]
            sample_rotation = rotation.cpu().numpy()[sample_idx]
            sample_sizes = [cell_sizes[i] for i in sample_idx]
            
            fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
            
            for r, c, size in zip(sample_rotation, sample_points, sample_sizes):
                S = np.array([
                    [size[0] * 0.5, 0, 0],
                    [0, size[1] * 0.5, 0],
                    [0, 0, 0]
                ])
                # Color based on cell size (smaller = red, larger = green)
                cell_area = size[0] * size[1]
                if cell_area < 0.01:
                    color = (1, 0, 0, 0.8)  # Red for fine cells
                elif cell_area < 0.05:
                    color = (1, 0.5, 0, 0.6)  # Orange for medium cells
                else:
                    color = (0, 1, 0, 0.4)  # Green for coarse cells
                
                plot_gaussion_3d(figure=fig, R=r, center=c, S=S, num=50, color=color, plot_axis=False)
            
            mlab.show()
        except Exception as e:
            print(f"可视化失败: {e}")
