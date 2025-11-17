"""
优化的 KD-tree 自适应网格实现 (V2 - 积分图加速版)

核心改进：
1. 预计算 BEV 特征图（几何 + 语义）
2. 使用积分图（DP）实现 O(1) 区域查询
3. 轨迹裁剪减少计算范围
4. 特征计算与树构建完全解耦

性能：
- 预处理：O(H×W) 构建 BEV 和积分图
- 树构建：O(M) 其中 M = 节点数，每次查询 O(1)
- 预期加速：30-100倍
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import time
import cv2


class KDTreeNode:
    """KD-tree 节点"""
    
    def __init__(self, min_coords: np.ndarray, max_coords: np.ndarray, depth: int = 0):
        self.min_coords = min_coords  # [x_min, y_min]
        self.max_coords = max_coords  # [x_max, y_max]
        self.depth = depth
        self.is_leaf = True
        self.children = []  # 子节点列表（最多4个）
        
    def get_center(self) -> np.ndarray:
        """获取节点中心"""
        return (self.min_coords + self.max_coords) / 2
    
    def get_area(self) -> float:
        """获取节点面积"""
        size = self.max_coords - self.min_coords
        return size[0] * size[1]
    
    def get_size(self) -> np.ndarray:
        """获取节点尺寸 [width, height]"""
        return self.max_coords - self.min_coords


class AdaptiveKDTreeV2:
    """
    基于积分图的高效 KD-tree 实现
    
    流程：
    1. 预处理：构建 BEV 特征图和积分图
    2. 构建树：递归查询积分图，O(1) 判断是否细分
    """
    
    def __init__(
        self,
        bev_resolution: float = 0.05,
        min_area: float = 0.0025,
        max_depth: int = 18,
        feature_threshold: float = None,  # None 表示使用自动阈值
        auto_threshold_percentile: float = 10.0,  # 自动阈值使用的分位数
        geo_weight: float = 0.7,
        sem_weight: float = 0.3,
    ):
        """
        初始化 KD-tree
        
        Args:
            bev_resolution: BEV 特征图分辨率 (m/pixel)
            min_area: 最小区域面积 (m²)
            max_depth: 最大树深度
            feature_threshold: 特征强度阈值，None 表示使用自动阈值
            auto_threshold_percentile: 自动阈值的分位数 (0-100)，越小越细分
            geo_weight: 几何特征权重
            sem_weight: 语义特征权重
        """
        self.bev_resolution = bev_resolution
        self.min_area = min_area
        self.max_depth = max_depth
        self.feature_threshold = feature_threshold
        self.auto_threshold_percentile = auto_threshold_percentile
        self.geo_weight = geo_weight
        self.sem_weight = sem_weight
        
        # 数据存储
        self.root = None
        self.leaves = []
        
        # BEV 特征图和积分图（预处理阶段构建）
        self.feature_map = None  # (H, W) 特征强度
        self.integral_map = None  # (H+1, W+1) 积分图
        self.bev_min_coords = None  # BEV 原点 [x_min, y_min]
        self.bev_max_coords = None  # BEV 最大坐标 [x_max, y_max]
        
        # 统计信息
        self.stats = {
            'num_nodes': 0,
            'num_leaves': 0,
            'max_depth_reached': 0,
            'preprocess_time': 0.0,
            'build_time': 0.0,
        }
    
    def build(
        self,
        boundary_points: np.ndarray,
        sem_feat: torch.Tensor,
        trajectory_mask: Optional[np.ndarray] = None,
        cut_range: float = 5.0,
        bev_min: Optional[np.ndarray] = None,
        bev_max: Optional[np.ndarray] = None
    ) -> 'AdaptiveKDTreeV2':
        """
        构建 KD-tree
        
        Args:
            boundary_points: 边界点 (N, 2)
            sem_feat: 语义特征 (N, C)
            trajectory_mask: 轨迹掩码 (H, W)，True 表示轨迹区域
            cut_range: 轨迹裁剪范围 (m)
            bev_min: 预定义的BEV最小坐标 [x_min, y_min]，如果提供则不使用轨迹掩码裁剪范围
            bev_max: 预定义的BEV最大坐标 [x_max, y_max]，如果提供则不使用轨迹掩码裁剪范围
        
        Returns:
            self
        """
        print("\n" + "="*60)
        print("开始构建 KD-tree (积分图优化版)")
        print("="*60)
        
        # 1. 预处理：构建 BEV 特征图和积分图
        self._preprocess(boundary_points, sem_feat, trajectory_mask, cut_range, bev_min, bev_max)
        
        # 2. 构建树：递归细分
        self._build_tree()
        
        # 3. 收集叶子节点
        self._collect_leaves()
        
        # 4. 打印统计信息
        self._print_stats()
        
        return self
    
    def _preprocess(
        self,
        boundary_points: np.ndarray,
        sem_feat: torch.Tensor,
        trajectory_mask: Optional[np.ndarray],
        cut_range: float,
        bev_min: Optional[np.ndarray] = None,
        bev_max: Optional[np.ndarray] = None
    ):
        """预处理：构建 BEV 特征图和积分图"""
        print("\n[1/3] 预处理阶段...")
        t_start = time.time()
        
        # 1.1 确定 BEV 范围
        if bev_min is not None and bev_max is not None:
            # 使用预定义的BEV范围（来自轨迹点+cut_range）
            print(f"  - 使用预定义的 BEV 范围（基于轨迹点）")
            self.bev_min_coords = bev_min
            self.bev_max_coords = bev_max
        elif trajectory_mask is not None:
            # 使用轨迹掩码裁剪（旧逻辑，会缩小范围）
            print(f"  - 使用轨迹掩码裁剪 BEV 范围 (cut_range={cut_range}m)")
            self.bev_min_coords, self.bev_max_coords = self._get_roi_from_trajectory(
                boundary_points, trajectory_mask, cut_range
            )
        else:
            print("  - 使用全部数据范围")
            self.bev_min_coords = boundary_points.min(axis=0)
            self.bev_max_coords = boundary_points.max(axis=0)
        
        bev_size = self.bev_max_coords - self.bev_min_coords
        print(f"  - BEV 范围: [{self.bev_min_coords[0]:.2f}, {self.bev_max_coords[0]:.2f}] x "
              f"[{self.bev_min_coords[1]:.2f}, {self.bev_max_coords[1]:.2f}]")
        print(f"  - BEV 尺寸: {bev_size[0]:.2f}m x {bev_size[1]:.2f}m")
        
        # 1.2 构建 BEV 特征图
        print(f"  - 构建 BEV 特征图 (分辨率={self.bev_resolution}m/pixel)...")
        self.feature_map = self._build_feature_map(boundary_points, sem_feat)
        print(f"    特征图尺寸: {self.feature_map.shape}")
        
        # 1.3 构建积分图
        print(f"  - 构建积分图...")
        self.integral_map = self._build_integral_map(self.feature_map)
        print(f"    积分图尺寸: {self.integral_map.shape}")
        
        # 1.4 确定阈值（如果未指定）
        if self.feature_threshold is None:
            # 使用分位数而不是均值（避免阈值过高）
            self.feature_threshold = float(np.percentile(self.feature_map, self.auto_threshold_percentile))
            print(f"  - 自动阈值（BEV {self.auto_threshold_percentile}分位数）: {self.feature_threshold:.10f}")
            print(f"    (BEV 均值: {np.mean(self.feature_map):.10f}, 用于参考)")
        else:
            print(f"  - 指定阈值: {self.feature_threshold:.10f}")
        
        self.stats['preprocess_time'] = time.time() - t_start
        print(f"  ✓ 预处理完成，耗时: {self.stats['preprocess_time']:.2f}s")
    
    def _get_roi_from_trajectory(
        self,
        boundary_points: np.ndarray,
        trajectory_mask: np.ndarray,
        cut_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从轨迹掩码获取 ROI"""
        # trajectory_mask 是布尔数组，True 表示轨迹区域
        # 找到所有 True 的位置
        y_indices, x_indices = np.where(trajectory_mask)
        
        if len(x_indices) == 0:
            # 如果没有轨迹，使用全部范围
            return boundary_points.min(axis=0), boundary_points.max(axis=0)
        
        # 将像素坐标转换为世界坐标
        # 假设 trajectory_mask 和 boundary_points 共享坐标系
        global_min = boundary_points.min(axis=0)
        global_max = boundary_points.max(axis=0)
        
        # 计算掩码分辨率
        mask_h, mask_w = trajectory_mask.shape
        x_coords = global_min[0] + x_indices * (global_max[0] - global_min[0]) / mask_w
        y_coords = global_min[1] + y_indices * (global_max[1] - global_min[1]) / mask_h
        
        # 添加 cut_range 裕度
        roi_min = np.array([x_coords.min() - cut_range, y_coords.min() - cut_range])
        roi_max = np.array([x_coords.max() + cut_range, y_coords.max() + cut_range])
        
        # 裁剪到有效范围
        roi_min = np.maximum(roi_min, global_min)
        roi_max = np.minimum(roi_max, global_max)
        
        return roi_min, roi_max
    
    def _build_feature_map(
        self,
        boundary_points: np.ndarray,
        sem_feat: torch.Tensor
    ) -> np.ndarray:
        """构建 BEV 特征图"""
        # 计算 BEV 网格尺寸
        bev_size = self.bev_max_coords - self.bev_min_coords
        grid_w = int(np.ceil(bev_size[0] / self.bev_resolution))
        grid_h = int(np.ceil(bev_size[1] / self.bev_resolution))
        
        # 初始化特征图
        geo_map = np.zeros((grid_h, grid_w), dtype=np.float32)  # 几何密度
        sem_map = np.zeros((grid_h, grid_w), dtype=np.float32)  # 语义熵
        count_map = np.zeros((grid_h, grid_w), dtype=np.int32)  # 点数
        
        # 将点分配到网格
        points_rel = boundary_points - self.bev_min_coords
        grid_x = (points_rel[:, 0] / self.bev_resolution).astype(np.int32)
        grid_y = (points_rel[:, 1] / self.bev_resolution).astype(np.int32)
        
        # 裁剪到有效范围
        valid_mask = (grid_x >= 0) & (grid_x < grid_w) & (grid_y >= 0) & (grid_y < grid_h)
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        sem_feat_valid = sem_feat[valid_mask]
        
        # 统计每个网格的点数和语义特征
        for i in range(len(grid_x)):
            x, y = grid_x[i], grid_y[i]
            count_map[y, x] += 1
            
            # 计算语义熵
            feat = sem_feat_valid[i]
            if isinstance(feat, torch.Tensor):
                probs = torch.softmax(feat, dim=0).cpu().numpy()
            else:
                # 如果是 numpy 数组，手动计算 softmax
                feat = np.array(feat)
                exp_feat = np.exp(feat - np.max(feat))
                probs = exp_feat / np.sum(exp_feat)
            
            probs = probs[probs > 1e-6]  # 避免 log(0)
            entropy = -np.sum(probs * np.log(probs))
            sem_map[y, x] += entropy
        
        # 计算几何强度（归一化点密度）
        # 使用 log(1 + density) 压缩动态范围
        density = count_map.astype(np.float32)
        geo_map = np.log(1 + density) / np.log(1 + 100)  # 归一化到 [0, 1]
        geo_map = np.clip(geo_map, 0, 1)
        
        # 计算语义强度（归一化熵）
        mask = count_map > 0
        sem_map[mask] /= count_map[mask]
        max_entropy = np.log(sem_feat.shape[1])  # log(num_classes)
        sem_map = np.clip(sem_map / max_entropy, 0, 1)
        
        # 合成特征图
        feature_map = self.geo_weight * geo_map + self.sem_weight * sem_map
        
        return feature_map
    
    def _build_integral_map(self, feature_map: np.ndarray) -> np.ndarray:
        """构建积分图（DP 图）"""
        H, W = feature_map.shape
        integral = np.zeros((H + 1, W + 1), dtype=np.float64)
        
        # 动态规划构建积分图
        # integral[i][j] = sum(feature_map[0:i, 0:j])
        for i in range(1, H + 1):
            for j in range(1, W + 1):
                integral[i, j] = (
                    feature_map[i-1, j-1]
                    + integral[i-1, j]
                    + integral[i, j-1]
                    - integral[i-1, j-1]
                )
        
        return integral
    
    def _query_region_feature(
        self,
        min_coords: np.ndarray,
        max_coords: np.ndarray
    ) -> float:
        """
        O(1) 查询矩形区域的平均特征强度
        
        Args:
            min_coords: [x_min, y_min] 世界坐标
            max_coords: [x_max, y_max] 世界坐标
        
        Returns:
            区域平均特征强度
        """
        # 转换为网格坐标
        rel_min = min_coords - self.bev_min_coords
        rel_max = max_coords - self.bev_min_coords
        
        x1 = int(np.floor(rel_min[0] / self.bev_resolution))
        y1 = int(np.floor(rel_min[1] / self.bev_resolution))
        x2 = int(np.ceil(rel_max[0] / self.bev_resolution))
        y2 = int(np.ceil(rel_max[1] / self.bev_resolution))
        
        # 裁剪到有效范围
        H, W = self.feature_map.shape
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        
        # 使用积分图计算区域和
        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            return 0.0
        
        region_sum = (
            self.integral_map[y2, x2]
            - self.integral_map[y1, x2]
            - self.integral_map[y2, x1]
            + self.integral_map[y1, x1]
        )
        
        return region_sum / area
    
    def _build_tree(self):
        """构建 KD-tree"""
        print("\n[2/3] 构建树...")
        t_start = time.time()
        
        # 创建根节点
        self.root = KDTreeNode(self.bev_min_coords, self.bev_max_coords, depth=0)
        self.stats['num_nodes'] = 1
        
        # 递归细分
        self._subdivide(self.root)
        
        self.stats['build_time'] = time.time() - t_start
        print(f"  ✓ 树构建完成，耗时: {self.stats['build_time']:.2f}s")
    
    def _subdivide(self, node: KDTreeNode):
        """递归细分节点"""
        # 终止条件
        if node.depth >= self.max_depth or node.get_area() <= self.min_area:
            self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], node.depth)
            return
        
        # 查询区域特征强度
        feature_intensity = self._query_region_feature(node.min_coords, node.max_coords)
        
        # 判断是否细分
        if feature_intensity <= self.feature_threshold:
            self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], node.depth)
            return
        
        # 细分为 4 个子节点
        center = node.get_center()
        child_coords = [
            (node.min_coords, center),                        # 左下
            (np.array([center[0], node.min_coords[1]]), np.array([node.max_coords[0], center[1]])),  # 右下
            (center, node.max_coords),                        # 右上
            (np.array([node.min_coords[0], center[1]]), np.array([center[0], node.max_coords[1]])),  # 左上
        ]
        
        node.is_leaf = False
        for min_c, max_c in child_coords:
            child = KDTreeNode(min_c, max_c, depth=node.depth + 1)
            node.children.append(child)
            self.stats['num_nodes'] += 1
            self._subdivide(child)
    
    def _collect_leaves(self):
        """收集所有叶子节点"""
        print("\n[3/3] 收集叶子节点...")
        self.leaves = []
        self._collect_leaves_recursive(self.root)
        self.stats['num_leaves'] = len(self.leaves)
        print(f"  ✓ 收集完成，叶子节点数: {self.stats['num_leaves']}")
    
    def _collect_leaves_recursive(self, node: KDTreeNode):
        """递归收集叶子节点"""
        if node.is_leaf:
            self.leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves_recursive(child)
    
    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("KD-tree 构建完成")
        print("="*60)
        print(f"总节点数: {self.stats['num_nodes']}")
        print(f"叶子节点数: {self.stats['num_leaves']}")
        print(f"最大深度: {self.stats['max_depth_reached']}")
        print(f"压缩比: {self.stats['num_leaves'] / (self.feature_map.shape[0] * self.feature_map.shape[1]):.2%}")
        print(f"预处理时间: {self.stats['preprocess_time']:.2f}s")
        print(f"树构建时间: {self.stats['build_time']:.2f}s")
        print(f"总时间: {self.stats['preprocess_time'] + self.stats['build_time']:.2f}s")
        print("="*60)
    
    def get_vertices(self) -> torch.Tensor:
        """获取所有叶子节点的顶点坐标"""
        if not self.leaves:
            return torch.empty((0, 2), dtype=torch.float32)
        
        vertices = []
        for leaf in self.leaves:
            # 每个叶子节点贡献 4 个顶点
            min_x, min_y = leaf.min_coords
            max_x, max_y = leaf.max_coords
            vertices.extend([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])
        
        return torch.tensor(vertices, dtype=torch.float32)
    
    def get_resolution_stats(self) -> Dict[str, float]:
        """获取分辨率统计"""
        if not self.leaves:
            return {}
        
        resolutions = []
        areas = []
        for leaf in self.leaves:
            size = leaf.get_size()
            resolutions.append(min(size[0], size[1]))
            areas.append(leaf.get_area())
        
        return {
            'min_resolution': float(np.min(resolutions)),
            'max_resolution': float(np.max(resolutions)),
            'mean_resolution': float(np.mean(resolutions)),
            'median_resolution': float(np.median(resolutions)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'mean_area': float(np.mean(areas)),
        }
