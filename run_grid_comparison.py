"""
完整对比流程：固定网格 vs 自适应网格 (KD-tree V2)

该脚本对比两种网格方案：
1. 固定网格：0.05m 分辨率
2. 自适应网格：基于积分图的 KD-tree，动态分辨率

所有日志保存到 Log 目录，按照时间和任务命名
"""

import os
import sys
import yaml
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
from addict import Addict

from datasets.nusc import NuscDataset
from utils.semantic_fusion import fuse_multiview_semantics
from utils.trajectory_utils import create_trajectory_mask
from models.kdtree_v2 import AdaptiveKDTreeV2


class Logger:
    """日志记录器，同时输出到控制台和文件"""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def load_configs(config_path: str):
    """加载配置文件"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return Addict(cfg)


def create_fixed_grid(boundary_points: np.ndarray, resolution: float = 0.05):
    """
    创建固定分辨率网格
    
    Args:
        boundary_points: (N, 2) 边界点
        resolution: 网格分辨率 (m)
    
    Returns:
        vertices: (M, 2) 网格顶点
        grid_shape: (H, W) 网格形状
    """
    min_coords = boundary_points.min(axis=0)
    max_coords = boundary_points.max(axis=0)
    
    # 计算网格数量
    grid_w = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    grid_h = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))
    
    # 生成网格顶点
    x = np.linspace(min_coords[0], min_coords[0] + grid_w * resolution, grid_w + 1)
    y = np.linspace(min_coords[1], min_coords[1] + grid_h * resolution, grid_h + 1)
    
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    return torch.tensor(vertices, dtype=torch.float32), (grid_h, grid_w)


def run_comparison(
    config_path: str = 'configs/local_nusc_mini.yaml',
    num_images: int = 120,
    fixed_resolution: float = 0.05,
    adaptive_resolution: float = 0.05,
    adaptive_min_area: float = 0.0025,
    adaptive_max_depth: int = 18,
    adaptive_percentile: float = 5.0,  # 更小的分位数，更多细分
):
    """
    运行完整对比流程
    
    Args:
        config_path: 数据集配置文件路径
        num_images: 使用的图像数量
        fixed_resolution: 固定网格分辨率
        adaptive_resolution: 自适应网格 BEV 分辨率
        adaptive_min_area: 自适应网格最小区域面积
        adaptive_max_depth: 自适应网格最大深度
        adaptive_percentile: 自适应网格自动阈值分位数
    """
    
    # 创建日志目录
    log_dir = Path("Log")
    log_dir.mkdir(exist_ok=True)
    
    # 创建日志文件名：时间_任务名.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_grid_comparison.log"
    
    # 重定向输出到日志文件
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print("=" * 80)
    print("完整对比流程：固定网格 vs 自适应网格")
    print("=" * 80)
    print(f"\n日志文件: {log_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 记录参数
    print("=" * 80)
    print("实验参数")
    print("=" * 80)
    print(f"数据集配置: {config_path}")
    print(f"使用图像数: {num_images}")
    print(f"\n固定网格:")
    print(f"  分辨率: {fixed_resolution}m")
    print(f"\n自适应网格:")
    print(f"  BEV 分辨率: {adaptive_resolution}m")
    print(f"  最小区域面积: {adaptive_min_area}m² ({np.sqrt(adaptive_min_area):.4f}m × {np.sqrt(adaptive_min_area):.4f}m)")
    print(f"  最大深度: {adaptive_max_depth}")
    print(f"  自动阈值分位数: {adaptive_percentile}%")
    print("=" * 80 + "\n")
    
    try:
        # ========== 步骤 1: 加载数据 ==========
        print("\n" + "=" * 80)
        print("步骤 1/6: 加载数据集")
        print("=" * 80)
        t_start = time.time()
        configs = load_configs(config_path)
        dataset = NuscDataset(configs.dataset, use_label=True, use_depth=False)
        load_time = time.time() - t_start
        print(f"  图像总数: {len(dataset.image_filenames_all)}")
        print(f"  耗时: {load_time:.2f}s")
        
        # ========== 步骤 2: 语义融合 ==========
        print("\n" + "=" * 80)
        print("步骤 2/6: 语义融合")
        print("=" * 80)
        t_start = time.time()
        enriched_pc = fuse_multiview_semantics(
            dataset,
            num_classes=65,
            use_all_cameras=True,
            max_images=num_images,
            verbose=True
        )
        fusion_time = time.time() - t_start
        
        xyz_all = enriched_pc["xyz"]
        sem_feat = enriched_pc["sem_feat"]
        boundary_points = xyz_all[:, :2]
        
        print(f"\n  使用图像: {num_images}")
        print(f"  道路点数: {len(xyz_all):,}")
        print(f"  语义特征: {sem_feat.shape}")
        print(f"  耗时: {fusion_time:.2f}s")
        
        # ========== 步骤 3: 轨迹提取 ==========
        print("\n" + "=" * 80)
        print("步骤 3/6: 提取车辆轨迹")
        print("=" * 80)
        t_start = time.time()
        
        min_coords = boundary_points.min(axis=0)
        max_coords = boundary_points.max(axis=0)
        trajectory_poses = dataset.chassis2world_unique[:, :2, 3]
        
        trajectory_mask, mask_shape = create_trajectory_mask(
            poses_xy=trajectory_poses,
            min_coords=min_coords,
            max_coords=max_coords,
            resolution=adaptive_resolution,
            cut_range=5.0
        )
        trajectory_time = time.time() - t_start
        print(f"  耗时: {trajectory_time:.2f}s")
        
        # ========== 步骤 4: 固定网格 ==========
        print("\n" + "=" * 80)
        print("步骤 4/6: 构建固定网格")
        print("=" * 80)
        t_start = time.time()
        fixed_vertices, fixed_shape = create_fixed_grid(boundary_points, fixed_resolution)
        fixed_time = time.time() - t_start
        
        fixed_num_cells = fixed_shape[0] * fixed_shape[1]
        fixed_num_vertices = len(fixed_vertices)
        
        print(f"  网格形状: {fixed_shape}")
        print(f"  网格数量: {fixed_num_cells:,}")
        print(f"  顶点数量: {fixed_num_vertices:,}")
        print(f"  耗时: {fixed_time:.2f}s")
        
        # ========== 步骤 5: 自适应网格 ==========
        print("\n" + "=" * 80)
        print("步骤 5/6: 构建自适应网格")
        print("=" * 80)
        
        tree = AdaptiveKDTreeV2(
            bev_resolution=adaptive_resolution,
            min_area=adaptive_min_area,
            max_depth=adaptive_max_depth,
            feature_threshold=None,
            auto_threshold_percentile=adaptive_percentile,
            geo_weight=0.7,
            sem_weight=0.3,
        )
        
        tree.build(
            boundary_points=boundary_points,
            sem_feat=sem_feat,
            trajectory_mask=trajectory_mask,
            cut_range=5.0
        )
        
        adaptive_vertices = tree.get_vertices()
        res_stats = tree.get_resolution_stats()
        
        # ========== 步骤 6: 结果对比 ==========
        print("\n" + "=" * 80)
        print("步骤 6/6: 结果对比与分析")
        print("=" * 80)
        
        print("\n【固定网格结果】")
        print(f"  分辨率: {fixed_resolution}m (固定)")
        print(f"  网格数量: {fixed_num_cells:,}")
        print(f"  顶点数量: {fixed_num_vertices:,}")
        print(f"  构建时间: {fixed_time:.2f}s")
        print(f"  存储大小: ~{fixed_num_vertices * 8 / 1024 / 1024:.2f} MB (float32)")
        
        print("\n【自适应网格结果】")
        print(f"  分辨率范围: [{res_stats['min_resolution']:.4f}m, {res_stats['max_resolution']:.4f}m]")
        print(f"  平均分辨率: {res_stats['mean_resolution']:.4f}m")
        print(f"  中位数分辨率: {res_stats['median_resolution']:.4f}m")
        print(f"  网格数量: {tree.stats['num_leaves']:,}")
        print(f"  顶点数量: {len(adaptive_vertices):,}")
        print(f"  最大深度: {tree.stats['max_depth_reached']}")
        print(f"  构建时间: {tree.stats['preprocess_time'] + tree.stats['build_time']:.2f}s")
        print(f"    - 预处理: {tree.stats['preprocess_time']:.2f}s")
        print(f"    - 树构建: {tree.stats['build_time']:.2f}s")
        print(f"  存储大小: ~{len(adaptive_vertices) * 8 / 1024 / 1024:.2f} MB (float32)")
        
        print("\n【对比分析】")
        grid_reduction = (1 - tree.stats['num_leaves'] / fixed_num_cells) * 100
        vertex_reduction = (1 - len(adaptive_vertices) / fixed_num_vertices) * 100
        time_ratio = (tree.stats['preprocess_time'] + tree.stats['build_time']) / fixed_time
        
        print(f"  网格减少: {grid_reduction:.2f}% ({fixed_num_cells:,} → {tree.stats['num_leaves']:,})")
        print(f"  顶点减少: {vertex_reduction:.2f}% ({fixed_num_vertices:,} → {len(adaptive_vertices):,})")
        print(f"  时间比率: {time_ratio:.2f}x (自适应/固定)")
        print(f"  分辨率提升: 最小 {fixed_resolution / res_stats['min_resolution']:.2f}x")
        
        # 检查是否达到最小分辨率
        if res_stats['min_resolution'] > np.sqrt(adaptive_min_area) * 1.1:
            print(f"\n  ⚠️  警告: 最小分辨率 ({res_stats['min_resolution']:.4f}m) 大于设定值 ({np.sqrt(adaptive_min_area):.4f}m)")
            print(f"      建议降低 auto_threshold_percentile 参数 (当前: {adaptive_percentile}%)")
        else:
            print(f"\n  ✓ 已达到目标最小分辨率")
        
        print("\n【完整流程时间】")
        total_time = load_time + fusion_time + trajectory_time + fixed_time + tree.stats['preprocess_time'] + tree.stats['build_time']
        print(f"  数据加载: {load_time:.2f}s")
        print(f"  语义融合: {fusion_time:.2f}s")
        print(f"  轨迹提取: {trajectory_time:.2f}s")
        print(f"  固定网格: {fixed_time:.2f}s")
        print(f"  自适应网格: {tree.stats['preprocess_time'] + tree.stats['build_time']:.2f}s")
        print(f"  总计: {total_time:.2f}s")
        
        print("\n" + "=" * 80)
        print("实验完成！")
        print("=" * 80)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志文件: {log_file}\n")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭日志
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__
        logger.close()


if __name__ == '__main__':
    # 运行对比实验
    run_comparison(
        config_path='configs/local_nusc_mini.yaml',
        num_images=120,
        fixed_resolution=0.05,
        adaptive_resolution=0.05,
        adaptive_min_area=0.0025,  # 0.05m × 0.05m
        adaptive_max_depth=18,
        adaptive_percentile=5.0,  # 5分位数，更激进的细分
    )
