"""
Adaptive Road class that integrates KD-tree for optimized road surface reconstruction.
This replaces the fixed grid approach with adaptive KD-tree subdivision.
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

from .kdtree import AdaptiveKDTree
from .road import cut_point_by_pose  # Import the original cutting function


class AdaptiveRoad:
    """
    Adaptive Road class using KD-tree for optimized mesh generation.
    Optimized for road-like elongated regions with strong directional characteristics.
    """
    
    def __init__(self, config, dataset, device='cuda:0', vis=False):
        self.device = device
        self.resolution = config["bev_resolution"]
        self.cut_range = config["cut_range"]
        
        # KD-tree specific configuration
        self.kd_config = {
            'min_area': 0.1,
            'max_depth': 8,
            'feature_threshold': 0.1,
            'min_points_per_region': 5
        }
        
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
        
        print(f"Adaptive Road - X length: {self.bev_x_length:.2f}m, Y length: {self.bev_y_length:.2f}m")

        # Use KD-tree to generate adaptive vertices
        vertices, four_indices = self._create_adaptive_mesh(min_coords, max_coords, all_pose_xyz, num_classes, vis)
        
        # Store the generated vertices and indices
        self.vertices = vertices  # (M, 3)
        self.four_indices = four_indices.to(device)
        
        # Create BEV camera
        self._create_bev_camera(min_coords, max_coords)

    def _create_adaptive_mesh(self, min_coords, max_coords, all_pose_xyz, num_classes, vis=False):
        """
        Create adaptive mesh using KD-tree approach
        
        Args:
            min_coords: Minimum coordinates [x_min, y_min, z_min]
            max_coords: Maximum coordinates [x_max, y_max, z_max]
            all_pose_xyz: All pose positions [N, 3]
            num_classes: Number of semantic classes
            
        Returns:
            Tuple of (vertices, four_indices)
        """
        # Extract boundary points (trajectory points for road)
        boundary_points = all_pose_xyz[:, :2]  # x, y coordinates
        
        # Initialize KD-tree
        kd_tree = AdaptiveKDTree(
            min_area=self.kd_config['min_area'],
            max_depth=self.kd_config['max_depth'],
            feature_threshold=self.kd_config['feature_threshold'],
            min_points_per_region=self.kd_config['min_points_per_region']
        )
        
        # Build KD-tree from boundary points
        kd_tree.build_from_boundary_points(boundary_points)
        
        print(f"KD-tree built with {len(boundary_points)} boundary points")
        
        # Generate adaptive vertices from KD-tree
        vertices, grid_shape, resolutions = kd_tree.get_adaptive_vertices()
        
        print(f"Generated {len(vertices)} adaptive vertices")
        
        if len(vertices) == 0:
            print("No vertices generated, falling back to regular grid")
            # Fallback to regular grid if KD-tree fails
            from .road import create_rect_vertices
            vertices, _, _ = create_rect_vertices(min_coords, max_coords, self.resolution)
            four_indices = torch.zeros((len(vertices), 4), dtype=torch.long)
            return vertices, four_indices
        
        # Filter vertices using pose-based cutting (adapted for KD-tree vertices)
        cut_vertices, four_indices = self._cut_adaptive_vertices_by_pose(
            vertices, kd_tree, all_pose_xyz[:, :2]  # Pass xy coordinates
        )
        
        print(f"After pose cutting: {len(cut_vertices)} vertices remain")
        
        # Initialize z-coordinates and rotations
        vertices_with_z = self._initialize_z_and_rotation(cut_vertices, all_pose_xyz)
        
        # Initialize RGB and labels
        self.rotation, self.rgb, self.label = self._initialize_features(
            vertices_with_z, all_pose_xyz, num_classes
        )
        
        # Visualize if requested
        if vis and len(vertices_with_z) > 0:
            self._visualize_adaptive_mesh(vertices_with_z, min_coords)
        
        return vertices_with_z, four_indices

    def _cut_adaptive_vertices_by_pose(self, adaptive_vertices, kd_tree, poses_xy):
        """
        Cut adaptive KD-tree vertices using poses
        
        Args:
            adaptive_vertices: Vertices from KD-tree [M, 3]
            kd_tree: The KD-tree object
            poses_xy: Pose positions [N, 2]
            
        Returns:
            Tuple of (cut_vertices, four_indices)
        """
        if len(adaptive_vertices) == 0:
            return adaptive_vertices, torch.empty((0, 4), dtype=torch.long)
        
        # Get 2D coordinates of vertices
        vertex_xy = adaptive_vertices[:, :2].cpu().numpy()
        
        # Create mask for vertices near poses
        cut_mask = np.zeros(len(vertex_xy), dtype=bool)
        
        for pose_xy in poses_xy:
            # Calculate distance to each vertex
            distances = np.linalg.norm(vertex_xy - pose_xy, axis=1)
            # Mark vertices within cut_range
            cut_mask |= distances <= self.cut_range
        
        # Filter vertices
        cut_vertices = adaptive_vertices[cut_mask]
        
        # Generate four indices for filtered vertices
        cut_indices = np.where(cut_mask)[0]
        n_cut = len(cut_indices)
        
        if n_cut == 0:
            return cut_vertices, torch.empty((0, 4), dtype=torch.long)
        
        # Create four neighbor indices using spatial relationships
        four_indices = self._generate_spatial_neighbors(vertex_xy[cut_mask], n_cut)
        
        return torch.from_numpy(cut_vertices).float(), four_indices

    def _generate_spatial_neighbors(self, cut_vertex_xy, n_vertices):
        """
        Generate four spatial neighbor indices for vertices
        
        Args:
            cut_vertex_xy: 2D coordinates of cut vertices [M, 2]
            n_vertices: Number of vertices
            
        Returns:
            Tensor of shape [M, 4] with neighbor indices
        """
        four_indices = np.zeros((n_vertices, 4), dtype=np.int64)
        
        for i in range(n_vertices):
            vertex_pos = cut_vertex_xy[i]
            neighbors = [i]  # Start with self-reference
            
            # Find nearest neighbors in each direction
            min_distances = [float('inf')] * 4  # left, right, down, up
            
            for j in range(n_vertices):
                if i == j:
                    continue
                    
                other_pos = cut_vertex_xy[j]
                dx = other_pos[0] - vertex_pos[0]
                dy = other_pos[1] - vertex_pos[1]
                
                # Determine direction and update nearest neighbor
                if abs(dx) > abs(dy):  # Horizontal direction
                    if dx < 0 and abs(dx) < min_distances[0]:  # Left
                        min_distances[0] = abs(dx)
                        neighbors[0] = j
                    elif dx > 0 and abs(dx) < min_distances[1]:  # Right
                        min_distances[1] = abs(dx)
                        neighbors[1] = j
                else:  # Vertical direction
                    if dy < 0 and abs(dy) < min_distances[2]:  # Down
                        min_distances[2] = abs(dy)
                        neighbors[2] = j
                    elif dy > 0 and abs(dy) < min_distances[3]:  # Up
                        min_distances[3] = abs(dy)
                        neighbors[3] = j
            
            # Ensure we have exactly 4 neighbors
            while len(neighbors) < 4:
                neighbors.append(i)  # Pad with self-reference
            
            four_indices[i] = neighbors[:4]
        
        return torch.from_numpy(four_indices).long()

    def _initialize_z_and_rotation(self, vertices, all_pose_xyz):
        """
        Initialize z-coordinates and rotations for vertices
        
        Args:
            vertices: Vertices [M, 3] (x, y, 0)
            all_pose_xyz: All pose positions [N, 3]
            
        Returns:
            Vertices with initialized z-coordinates [M, 3]
        """
        # Convert to device
        vertices = vertices.to(self.device)
        traj_point = torch.from_numpy(all_pose_xyz).float().to(self.device)
        traj_rotation = torch.from_numpy(all_poses[:, :3, :3]).float().to(self.device)
        
        # Find nearest trajectory points for z and rotation assignment
        nearest_result = knn_points(vertices[None, :, :2], traj_point[None, :, :2], K=1)
        near_points = knn_gather(traj_point[None], nearest_result.idx)[0]  # (M, K, 3)
        
        nearest_idx = nearest_result.idx[0, :, 0]
        init_z = torch.mean(near_points, dim=1)[:, 2]  # (M,)
        vertices[:, 2] = init_z
        
        return vertices

    def _initialize_features(self, vertices, all_pose_xyz, num_classes):
        """
        Initialize rotation, RGB, and label features
        
        Args:
            vertices: Vertices [M, 3]
            all_pose_xyz: All pose positions [N, 3]
            num_classes: Number of semantic classes
            
        Returns:
            Tuple of (rotation, rgb, label)
        """
        all_poses = getattr(self, 'all_poses', None)
        if all_poses is None:
            # This would need to be passed or stored from the dataset
            all_poses = np.random.rand(*all_pose_xyz.shape)  # Placeholder
            
        traj_point = torch.from_numpy(all_pose_xyz).float().to(self.device)
        traj_rotation = torch.from_numpy(all_poses[:, :3, :3]).float().to(self.device)
        
        # Get nearest indices
        nearest_result = knn_points(vertices[None, :, :2], traj_point[None, :, :2], K=1)
        nearest_idx = nearest_result.idx[0, :, 0]
        
        # Initialize features
        rotation = traj_rotation[nearest_idx]  # (M, 3, 3)
        rotation_quat = matrix_to_quaternion(rotation)  # (M, 4)
        
        rgb = torch.zeros_like(vertices, device=self.device)
        label = torch.zeros((vertices.shape[0], num_classes), dtype=torch.float32, device=self.device)
        
        return rotation_quat, rgb, label

    def _create_bev_camera(self, min_coords, max_coords):
        """Create BEV orthographic camera"""
        SLACK_Z = 1
        mid_xy = (min_coords[:2] + max_coords[:2]) / 2
        bevcam2world = np.array([[1, 0, 0, mid_xy[0]],
                                 [0, -1, 0, mid_xy[1]],
                                 [0, 0, -1, self.max_z + SLACK_Z],
                                 [0, 0, 0, 1]])
        bevcam2world = torch.from_numpy(bevcam2world).float()

        render_resolution = self.resolution
        width = int(self.bev_x_length / render_resolution) + 1
        height = int(self.bev_y_length / render_resolution) + 1
        self.bev_camera = OrthographicCamera(
            R=bevcam2world[:3, :3], T=bevcam2world[:3, 3], W=width, H=height, znear=0,
            zfar=bevcam2world[2, 3] - self.min_z + SLACK_Z,
            top=-self.bev_y_length * 0.5, bottom=self.bev_y_length * 0.5, 
            right=self.bev_x_length * 0.5, left=-self.bev_x_length * 0.5, 
            device=self.device
        )

    def _visualize_adaptive_mesh(self, vertices, min_coords):
        """Visualize the adaptive mesh structure"""
        try:
            points = vertices.cpu().numpy()
            sample_idx = random.sample(range(points.shape[0]), min(150, len(points)))
            sample_points = points[sample_idx]

            fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
            mlab.points3d(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                         scale_factor=self.resolution * 0.5, color=(0.5, 0.5, 0.5), figure=fig)

            mlab.title('Adaptive KD-tree Road Mesh')
            mlab.show()
        except Exception as e:
            print(f"Visualization failed: {e}")

    def get_kd_tree_info(self):
        """
        Get information about the KD-tree structure
        
        Returns:
            Dictionary with KD-tree statistics
        """
        return {
            'num_vertices': len(self.vertices),
            'bev_x_length': self.bev_x_length,
            'bev_y_length': self.bev_y_length,
            'kd_config': self.kd_config
        }
