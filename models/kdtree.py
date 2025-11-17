"""
KD-tree implementation for adaptive road surface reconstruction.
Optimized for road-like elongated regions with strong directionality.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import cv2


class KDTreeNode:
    """KD-tree node representing a rectangular region"""
    
    def __init__(self, min_coords: np.ndarray, max_coords: np.ndarray, depth: int = 0):
        """
        Initialize KD-tree node
        
        Args:
            min_coords: Minimum coordinates [x_min, y_min]
            max_coords: Maximum coordinates [x_max, y_max] 
            depth: Current depth in the tree
        """
        self.min_coords = min_coords.copy()
        self.max_coords = max_coords.copy()
        self.depth = depth
        self.left = None
        self.right = None
        self.center = (self.min_coords + self.max_coords) / 2.0
        self.area = self._calculate_area()
        self.is_leaf = True
        self.feature_value = 0.0
        
    def _calculate_area(self) -> float:
        """Calculate the area of the rectangular region"""
        return float(np.prod(self.max_coords - self.min_coords))
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is within this region"""
        return np.all((point >= self.min_coords) & (point <= self.max_coords))
    
    def get_region_size(self) -> Tuple[float, float]:
        """Get the size of the region (length, width)"""
        size = self.max_coords - self.min_coords
        return float(size[0]), float(size[1])  # (x_length, y_length)
    
    def should_subdivide(self, min_area: float = 0.1, max_depth: int = 8) -> bool:
        """
        Determine if this node should be subdivided based on adaptive criteria
        
        Args:
            min_area: Minimum area threshold to stop subdivision
            max_depth: Maximum depth to prevent over-refinement
        """
        if self.depth >= max_depth:
            return False
        if self.area <= min_area:
            return False
        return True
    
    def subdivide(self, feature_threshold: float = 0.1) -> bool:
        """
        Subdivide this node into two children based on region aspect ratio
        
        Args:
            feature_threshold: Threshold for feature value difference to justify subdivision
            
        Returns:
            bool: True if subdivision occurred, False otherwise
        """
        if not self.should_subdivide():
            return False
            
        length, width = self.get_region_size()
        
        # For road-like regions: if length > width, split along length (x-axis)
        # This adapts to the directional nature of roads
        if length > width:
            # Split along x-axis (longitudinal direction)
            split_coord = self.center[0]
            
            left_min = self.min_coords.copy()
            left_max = np.array([split_coord, self.max_coords[1]])
            
            right_min = np.array([split_coord, self.min_coords[1]])
            right_max = self.max_coords.copy()
            
        else:
            # Split along y-axis for square or width-dominant regions
            split_coord = self.center[1]
            
            left_min = self.min_coords.copy()
            left_max = np.array([self.max_coords[0], split_coord])
            
            right_min = np.array([self.min_coords[0], split_coord])
            right_max = self.max_coords.copy()
        
        # Create child nodes
        self.left = KDTreeNode(left_min, left_max, self.depth + 1)
        self.right = KDTreeNode(right_min, right_max, self.depth + 1)
        self.is_leaf = False
        
        return True
    
    def subdivide_with_params(self, min_area: float = 0.1, max_depth: int = 8) -> bool:
        """
        Subdivide this node with explicit parameters
        
        Args:
            min_area: Minimum area threshold
            max_depth: Maximum depth
            
        Returns:
            bool: True if subdivision occurred, False otherwise
        """
        if not self.should_subdivide(min_area=min_area, max_depth=max_depth):
            return False
            
        length, width = self.get_region_size()
        
        # For road-like regions: if length > width, split along length (x-axis)
        if length > width:
            # Split along x-axis (longitudinal direction)
            split_coord = self.center[0]
            
            left_min = self.min_coords.copy()
            left_max = np.array([split_coord, self.max_coords[1]])
            
            right_min = np.array([split_coord, self.min_coords[1]])
            right_max = self.max_coords.copy()
            
        else:
            # Split along y-axis
            split_coord = self.center[1]
            
            left_min = self.min_coords.copy()
            left_max = np.array([self.max_coords[0], split_coord])
            
            right_min = np.array([self.min_coords[0], split_coord])
            right_max = self.max_coords.copy()
        
        # Create child nodes
        self.left = KDTreeNode(left_min, left_max, self.depth + 1)
        self.right = KDTreeNode(right_min, right_max, self.depth + 1)
        self.is_leaf = False
        
        return True
    
    def get_all_leaf_centers(self) -> List[np.ndarray]:
        """Get all leaf node centers"""
        if self.is_leaf:
            return [self.center.copy()]
        
        centers = []
        if self.left:
            centers.extend(self.left.get_all_leaf_centers())
        if self.right:
            centers.extend(self.right.get_all_leaf_centers())
        return centers
    
    def get_leaf_regions(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all leaf regions as (min_coords, max_coords) tuples"""
        if self.is_leaf:
            return [(self.min_coords.copy(), self.max_coords.copy())]
        
        regions = []
        if self.left:
            regions.extend(self.left.get_leaf_regions())
        if self.right:
            regions.extend(self.right.get_leaf_regions())
        return regions


class AdaptiveKDTree:
    """
    Adaptive KD-tree for road surface reconstruction.
    Optimized for elongated road regions with strong directional characteristics.
    """
    
    def __init__(self, min_area: float = 0.1, max_depth: int = 8, 
                 feature_threshold: float = 0.1, min_points_per_region: int = 5):
        """
        Initialize adaptive KD-tree
        
        Args:
            min_area: Minimum area threshold to stop subdivision
            max_depth: Maximum depth of the tree
            feature_threshold: Feature difference threshold for subdivision decision
            min_points_per_region: Minimum points required per region
        """
        self.min_area = min_area
        self.max_depth = max_depth
        self.feature_threshold = feature_threshold
        self.min_points_per_region = min_points_per_region
        self.root = None
        self.boundary_points = None
        self.feature_map = None
        
    def build_from_boundary_points(self, boundary_points: np.ndarray, 
                                 feature_map: Optional[np.ndarray] = None):
        """
        Build KD-tree from boundary points with adaptive subdivision
        
        Args:
            boundary_points: Array of boundary points [N, 2]
            feature_map: Optional 2D feature map for region analysis
        """
        self.boundary_points = boundary_points
        self.feature_map = feature_map
        
        # Calculate bounding box from boundary points
        min_coords = np.min(boundary_points, axis=0)
        max_coords = np.max(boundary_points, axis=0)
        
        # Create root node
        self.root = KDTreeNode(min_coords, max_coords, depth=0)
        
        # Recursively build tree with adaptive subdivision
        self._build_recursive(self.root)
    
    def _build_recursive(self, node: KDTreeNode):
        """
        Recursively build KD-tree with adaptive subdivision based on features
        
        Args:
            node: Current node to process
        """
        # 首先检查基本条件（深度和面积）
        if not node.should_subdivide(min_area=self.min_area, max_depth=self.max_depth):
            return
        
        # Calculate feature statistics for this region
        feature_stats = self._calculate_region_features(node)
        node.feature_value = feature_stats['variance']
        
        # Subdivision decision based on feature complexity
        if self._should_subdivide_by_features(node, feature_stats):
            # Subdivide the node（传递正确的参数）
            if node.subdivide_with_params(min_area=self.min_area, max_depth=self.max_depth):
                # Recursively process children
                self._build_recursive(node.left)
                self._build_recursive(node.right)
    
    def _calculate_region_features(self, node: KDTreeNode) -> dict:
        """
        Calculate feature statistics for a region
        
        Args:
            node: KD-tree node representing the region
            
        Returns:
            dict: Feature statistics including variance, mean, etc.
        """
        # Get points within this region
        region_points = self._get_points_in_region(node)
        
        if len(region_points) < 2:
            return {'variance': 0.0, 'mean': 0.0, 'point_count': len(region_points)}
        
        # Calculate spatial variance
        variance = np.var(region_points, axis=0)
        feature_variance = np.mean(variance)  # Combined variance
        
        # If feature map is available, calculate feature-based statistics
        if self.feature_map is not None:
            feature_variance += self._calculate_feature_variance(node)
        
        return {
            'variance': feature_variance,
            'mean': np.mean(region_points, axis=0),
            'point_count': len(region_points),
            'spatial_variance': variance
        }
    
    def _get_points_in_region(self, node: KDTreeNode) -> np.ndarray:
        """
        Get boundary points within a given region
        
        Args:
            node: KD-tree node representing the region
            
        Returns:
            Array of points within the region
        """
        if self.boundary_points is None:
            return np.array([])
        
        # Create mask for points within region bounds
        mask = np.all((self.boundary_points >= node.min_coords) & 
                     (self.boundary_points <= node.max_coords), axis=1)
        
        return self.boundary_points[mask]
    
    def _calculate_feature_variance(self, node: KDTreeNode) -> float:
        """
        Calculate feature variance from feature map within region
        
        This computes semantic complexity using entropy and variance of semantic features.
        High entropy indicates mixed semantic classes, suggesting the region should be subdivided.
        
        Args:
            node: KD-tree node representing the region
            
        Returns:
            Feature variance value (higher = more semantic complexity)
        """
        if self.feature_map is None:
            return 0.0
        
        # Get mask for points in this region
        if self.boundary_points is None:
            return 0.0
            
        mask = np.all((self.boundary_points >= node.min_coords) & 
                     (self.boundary_points <= node.max_coords), axis=1)
        
        if not np.any(mask):
            return 0.0
        
        # Get semantic features for points in this region
        region_features = self.feature_map[mask]  # (M, C) where C is number of classes
        
        if len(region_features) < 2:
            return 0.0
        
        # Calculate semantic complexity metrics
        
        # 1. Semantic entropy: measures class diversity
        # Average feature distribution in this region
        mean_features = np.mean(region_features, axis=0)  # (C,)
        mean_features = mean_features / (np.sum(mean_features) + 1e-8)  # Normalize
        
        # Shannon entropy: -sum(p * log(p))
        entropy = -np.sum(mean_features * np.log(mean_features + 1e-8))
        
        # 2. Feature variance: measures inconsistency between points
        feature_std = np.std(region_features, axis=0)  # (C,)
        variance_score = np.mean(feature_std)
        
        # 3. Dominant class confidence: low confidence suggests subdivision
        max_prob = np.max(mean_features)
        confidence_penalty = 1.0 - max_prob  # Higher penalty if no dominant class
        
        # Combine metrics (weighted sum)
        # Higher values indicate more semantic complexity requiring subdivision
        semantic_complexity = (
            0.5 * entropy +           # Entropy contribution
            0.3 * variance_score +    # Variance contribution
            0.2 * confidence_penalty  # Confidence penalty
        )
        
        return float(semantic_complexity)
    
    def _should_subdivide_by_features(self, node: KDTreeNode, feature_stats: dict) -> bool:
        """
        Determine if subdivision should occur based on feature analysis
        
        混合策略：几何密度 + 语义复杂度 + 灵活点数限制
        目标：复杂区域细分到 min_area，简单区域早点停止
        
        Args:
            node: KD-tree node
            feature_stats: Feature statistics for the region
            
        Returns:
            bool: True if subdivision should occur
        """
        point_count = feature_stats['point_count']
        
        # 避免除零
        if point_count == 0:
            return False
        
        # 计算点密度（点/m²）
        point_density = point_count / max(node.area, 1e-6)
        
        # 条件 1：高语义复杂度（必须细分）
        if feature_stats['variance'] > self.feature_threshold:
            if point_count >= 5:  # 最少 5 个点即可
                return True
        
        # 条件 2：高点密度区域（几何复杂，如路口、车道线密集处）
        if point_density > 20:  # 每平方米 > 20 个点
            if point_count >= 10:
                return True
        
        # 条件 3：中等密度 + 区域还很大
        if point_density > 5 and node.area > self.min_area * 4:
            if point_count >= 10:
                return True
        
        # 条件 4：允许细分接近 min_area（无论语义和密度）
        # 这确保即使在中等密度区域也能达到较细的分辨率
        if node.area > self.min_area * 2:
            if point_count >= 3:  # 极低点数要求
                return True
        
        return False
    
    def get_adaptive_vertices(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        """
        Generate vertices from KD-tree leaf nodes
        
        Returns:
            Tuple of (vertices, grid_shape, resolutions)
            - vertices: Array of vertex positions [M, 3]
            - grid_shape: Shape of the logical grid (x_res, y_res)
            - resolutions: (x_resolution, y_resolution)
        """
        if self.root is None:
            return np.array([]), (0, 0), (0.0, 0.0)
        
        # Get all leaf regions
        leaf_regions = self.root.get_leaf_regions()
        
        if not leaf_regions:
            return np.array([]), (0, 0), (0.0, 0.0)
        
        vertices = []
        
        # Generate vertices at leaf region centers
        for min_coords, max_coords in leaf_regions:
            center = (min_coords + max_coords) / 2.0
            
            # Add z-coordinate (initially 0, will be set later)
            vertex = np.array([center[0], center[1], 0.0])
            vertices.append(vertex)
        
        vertices = np.array(vertices)
        
        # Calculate approximate grid shape and resolutions
        if len(vertices) > 0:
            # Simple estimation based on bounding box and vertex count
            root_area = self.root.area
            avg_area_per_vertex = root_area / len(vertices)
            avg_resolution = np.sqrt(avg_area_per_vertex)
            
            # Estimate grid dimensions
            x_range = self.root.max_coords[0] - self.root.min_coords[0]
            y_range = self.root.max_coords[1] - self.root.min_coords[1]
            
            x_res = max(1, int(x_range / avg_resolution))
            y_res = max(1, int(y_range / avg_resolution))
            
            return vertices, (x_res, y_res), (avg_resolution, avg_resolution)
        
        return vertices, (0, 0), (0.0, 0.0)
    
    def get_four_indices(self, all_vertices: np.ndarray) -> np.ndarray:
        """
        Generate four neighbor indices for each vertex (compatibility with original code)
        
        Args:
            all_vertices: All generated vertices
            
        Returns:
            Array of shape [N, 4] with neighbor indices
        """
        n_vertices = len(all_vertices)
        four_indices = np.arange(n_vertices).reshape(-1, 1).repeat(4, axis=1)
        
        # For simplicity, use sequential indexing (can be improved with spatial indexing)
        for i in range(n_vertices):
            four_indices[i, 0] = max(0, i - 1)  # left
            four_indices[i, 1] = min(n_vertices - 1, i + 1)  # right  
            four_indices[i, 2] = max(0, i - 10)  # up (assuming 10x10 grid for simplicity)
            four_indices[i, 3] = min(n_vertices - 1, i + 10)  # down
        
        return torch.from_numpy(four_indices).long()
    
    def visualize_tree(self, save_path: Optional[str] = None):
        """
        Visualize the KD-tree structure and subdivision
        
        Args:
            save_path: Optional path to save the visualization
        """
        if self.root is None:
            return
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot boundary points
        if self.boundary_points is not None:
            ax.scatter(self.boundary_points[:, 0], self.boundary_points[:, 1], 
                      c='black', s=1, alpha=0.5, label='Boundary Points')
        
        # Draw tree structure
        self._draw_node_recursive(self.root, ax)
        
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Adaptive KD-tree Structure')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _draw_node_recursive(self, node: KDTreeNode, ax):
        """Recursively draw tree nodes"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Draw current region
        rect = patches.Rectangle(node.min_coords, 
                           node.max_coords[0] - node.min_coords[0],
                           node.max_coords[1] - node.min_coords[1],
                           fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(rect)
        
        # Draw center point
        ax.plot(node.center[0], node.center[1], 'ro', markersize=3)
        
        # Recursively draw children
        if node.left:
            self._draw_node_recursive(node.left, ax)
        if node.right:
            self._draw_node_recursive(node.right, ax)
