"""
可视化局部区域的网格分布对比

展示固定网格 vs 自适应网格在 2m×2m 区域内的分布情况
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import yaml
from addict import Addict
from pathlib import Path
from datetime import datetime

import matplotlib as mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
mpl.rcParams['axes.unicode_minus'] = False


import sys

from datasets.nusc import NuscDataset
from utils.semantic_fusion import fuse_multiview_semantics
from utils.trajectory_utils import create_trajectory_mask
from models.kdtree_v2 import AdaptiveKDTreeV2


def create_fixed_grid_cells(min_coords, max_coords, resolution=0.05):
    """创建固定网格的单元格边界"""
    x = np.arange(min_coords[0], max_coords[0] + resolution, resolution)
    y = np.arange(min_coords[1], max_coords[1] + resolution, resolution)
    return x, y


def visualize_grid_comparison(
    boundary_points,
    sem_feat,
    trajectory_mask,
    output_dir="output/grid_visualization",
    region_size=2.0,  # 可视化区域大小 (m)
    fixed_resolution=0.05,
):
    """
    可视化固定网格和自适应网格的对比
    
    Args:
        boundary_points: (N, 2) 边界点
        sem_feat: (N, C) 语义特征
        trajectory_mask: (H, W) 轨迹掩码
        output_dir: 输出目录
        region_size: 可视化区域大小 (m)
        fixed_resolution: 固定网格分辨率
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("局部区域网格可视化")
    print("="*80)
    
    # 1. 构建自适应网格
    print("\n[1/3] 构建自适应网格...")
    tree = AdaptiveKDTreeV2(
        bev_resolution=0.05,
        min_area=0.0025,
        max_depth=18,
        auto_threshold_percentile=5.0,
        geo_weight=0.7,
        sem_weight=0.3,
    )
    
    tree.build(
        boundary_points=boundary_points,
        sem_feat=sem_feat,
        trajectory_mask=trajectory_mask,
        cut_range=5.0
    )
    
    # 2. 选择感兴趣区域 (ROI)
    print("\n[2/3] 选择可视化区域...")
    
    # 选择轨迹中间附近的区域
    min_coords = boundary_points.min(axis=0)
    max_coords = boundary_points.max(axis=0)
    center = (min_coords + max_coords) / 2
    
    # 选择3个不同的区域进行可视化
    regions = [
        {
            'name': 'center',
            'title': '中心区域（轨迹附近）',
            'center': center,
        },
        {
            'name': 'complex',
            'title': '复杂区域（高密度点）',
            'center': center + np.array([10.0, 5.0]),
        },
        {
            'name': 'simple',
            'title': '简单区域（低密度点）',
            'center': center + np.array([-10.0, -5.0]),
        },
    ]
    
    # 3. 为每个区域创建可视化
    print("\n[3/3] 生成可视化...")
    
    for region in regions:
        roi_center = region['center']
        roi_min = roi_center - region_size / 2
        roi_max = roi_center + region_size / 2
        
        print(f"\n  处理 {region['title']}...")
        print(f"    ROI: [{roi_min[0]:.2f}, {roi_max[0]:.2f}] x [{roi_min[1]:.2f}, {roi_max[1]:.2f}]")
        
        # 过滤 ROI 内的点
        mask = (
            (boundary_points[:, 0] >= roi_min[0]) &
            (boundary_points[:, 0] <= roi_max[0]) &
            (boundary_points[:, 1] >= roi_min[1]) &
            (boundary_points[:, 1] <= roi_max[1])
        )
        roi_points = boundary_points[mask]
        
        print(f"    包含点数: {len(roi_points):,}")
        
        # 过滤 ROI 内的自适应网格叶子节点
        roi_leaves = []
        for leaf in tree.leaves:
            # 检查叶子节点是否与 ROI 相交
            if (leaf.min_coords[0] < roi_max[0] and leaf.max_coords[0] > roi_min[0] and
                leaf.min_coords[1] < roi_max[1] and leaf.max_coords[1] > roi_min[1]):
                roi_leaves.append(leaf)
        
        print(f"    自适应网格数: {len(roi_leaves)}")
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # --- 子图 1: 点云分布 ---
        ax = axes[0]
        if len(roi_points) > 0:
            ax.scatter(roi_points[:, 0], roi_points[:, 1], 
                      s=1, c='gray', alpha=0.5, label='路面点')
        ax.set_xlim(roi_min[0], roi_max[0])
        ax.set_ylim(roi_min[1], roi_max[1])
        ax.set_aspect('equal')
        ax.set_title('路面点云分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # --- 子图 2: 固定网格 ---
        ax = axes[1]
        if len(roi_points) > 0:
            ax.scatter(roi_points[:, 0], roi_points[:, 1], 
                      s=1, c='gray', alpha=0.3)
        
        # 绘制固定网格线
        x_lines = np.arange(roi_min[0], roi_max[0] + fixed_resolution, fixed_resolution)
        y_lines = np.arange(roi_min[1], roi_max[1] + fixed_resolution, fixed_resolution)
        
        for x in x_lines:
            ax.axvline(x, color='blue', linewidth=0.5, alpha=0.6)
        for y in y_lines:
            ax.axhline(y, color='blue', linewidth=0.5, alpha=0.6)
        
        fixed_grid_count = len(x_lines) * len(y_lines)
        
        ax.set_xlim(roi_min[0], roi_max[0])
        ax.set_ylim(roi_min[1], roi_max[1])
        ax.set_aspect('equal')
        ax.set_title(f'固定网格 ({fixed_resolution}m)\n网格数: {fixed_grid_count}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.2)
        
        # --- 子图 3: 自适应网格 ---
        ax = axes[2]
        if len(roi_points) > 0:
            ax.scatter(roi_points[:, 0], roi_points[:, 1], 
                      s=1, c='gray', alpha=0.3)
        
        # 绘制自适应网格
        for leaf in roi_leaves:
            # 计算与 ROI 的交集
            cell_min = np.maximum(leaf.min_coords, roi_min)
            cell_max = np.minimum(leaf.max_coords, roi_max)
            
            width = cell_max[0] - cell_min[0]
            height = cell_max[1] - cell_min[1]
            
            if width > 0 and height > 0:
                # 根据网格大小设置颜色（越小越红）
                cell_area = leaf.get_area()
                if cell_area < 0.01:  # < 0.1m × 0.1m
                    color = 'red'
                    alpha = 0.3
                elif cell_area < 0.05:  # < 0.22m × 0.22m
                    color = 'orange'
                    alpha = 0.25
                else:
                    color = 'green'
                    alpha = 0.15
                
                rect = Rectangle(
                    (cell_min[0], cell_min[1]),
                    width, height,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha
                )
                ax.add_patch(rect)
        
        ax.set_xlim(roi_min[0], roi_max[0])
        ax.set_ylim(roi_min[1], roi_max[1])
        ax.set_aspect('equal')
        ax.set_title(f'自适应网格 (KD-tree)\n网格数: {len(roi_leaves)}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.2)
        
        # 添加图例
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='细分区域 (<0.1m)')
        orange_patch = mpatches.Patch(color='orange', alpha=0.25, label='中等区域 (0.1-0.22m)')
        green_patch = mpatches.Patch(color='green', alpha=0.15, label='稀疏区域 (>0.22m)')
        ax.legend(handles=[red_patch, orange_patch, green_patch], loc='upper right')
        
        # 整体标题
        plt.suptitle(f'{region["title"]} - 网格对比 ({region_size}m × {region_size}m)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = output_dir / f"grid_comparison_{region['name']}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ 保存: {save_path}")
        plt.close()
    
    # 4. 创建统计对比图
    print("\n  生成统计对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 分辨率分布直方图
    ax = axes[0, 0]
    resolutions = []
    for leaf in tree.leaves:
        size = leaf.get_size()
        resolutions.append(min(size[0], size[1]))
    
    ax.hist(resolutions, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(fixed_resolution, color='red', linestyle='--', linewidth=2, label=f'固定网格 ({fixed_resolution}m)')
    ax.set_xlabel('网格分辨率 (m)')
    ax.set_ylabel('数量')
    ax.set_title('自适应网格分辨率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 网格面积分布
    ax = axes[0, 1]
    areas = [leaf.get_area() for leaf in tree.leaves]
    ax.hist(areas, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(fixed_resolution**2, color='red', linestyle='--', linewidth=2, 
               label=f'固定网格 ({fixed_resolution**2:.4f}m²)')
    ax.set_xlabel('网格面积 (m²)')
    ax.set_ylabel('数量')
    ax.set_title('自适应网格面积分布')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 深度分布
    ax = axes[1, 0]
    depths = []
    def get_leaf_depths(node, depth=0):
        if node.is_leaf:
            depths.append(depth)
        else:
            for child in node.children:
                get_leaf_depths(child, depth + 1)
    get_leaf_depths(tree.root)
    
    ax.hist(depths, bins=range(max(depths)+2), color='lightgreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('树深度')
    ax.set_ylabel('叶子节点数量')
    ax.set_title('KD-tree 深度分布')
    ax.grid(True, alpha=0.3)
    
    # 网格数量对比
    ax = axes[1, 1]
    res_stats = tree.get_resolution_stats()
    
    # 计算固定网格数量
    bev_size = max_coords - min_coords
    fixed_grid_count = int(bev_size[0] / fixed_resolution) * int(bev_size[1] / fixed_resolution)
    
    categories = ['固定网格', '自适应网格']
    counts = [fixed_grid_count, tree.stats['num_leaves']]
    colors = ['lightcoral', 'lightblue']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('网格数量')
    ax.set_title('网格数量对比')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/fixed_grid_count*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('自适应网格统计分析', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / "grid_statistics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {save_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("可视化完成！")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print(f"生成文件:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")


def main():
    """主函数"""
    
    print("="*80)
    print("网格分布可视化工具")
    print("="*80)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    config_path = 'configs/local_nusc_mini.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    configs = Addict(cfg)
    
    dataset = NuscDataset(configs.dataset, use_label=True, use_depth=False)
    print("  ✓ 数据加载完成")
    
    # 语义融合
    print("\n[2/4] 语义融合...")
    enriched_pc = fuse_multiview_semantics(
        dataset,
        num_classes=65,
        use_all_cameras=True,
        max_images=120,
        verbose=False
    )
    
    xyz_all = enriched_pc["xyz"]
    sem_feat = enriched_pc["sem_feat"]
    boundary_points = xyz_all[:, :2]
    print(f"  ✓ 完成 ({len(boundary_points):,} 点)")
    
    # 轨迹掩码
    print("\n[3/4] 创建轨迹掩码...")
    min_coords = boundary_points.min(axis=0)
    max_coords = boundary_points.max(axis=0)
    trajectory_poses = dataset.chassis2world_unique[:, :2, 3]
    
    trajectory_mask, _ = create_trajectory_mask(
        poses_xy=trajectory_poses,
        min_coords=min_coords,
        max_coords=max_coords,
        resolution=0.05,
        cut_range=5.0
    )
    print("  ✓ 掩码创建完成")
    
    # 生成可视化
    print("\n[4/4] 生成可视化...")
    visualize_grid_comparison(
        boundary_points=boundary_points,
        sem_feat=sem_feat,
        trajectory_mask=trajectory_mask,
        region_size=2.0,  # 2m × 2m 区域
        fixed_resolution=0.05,
    )


if __name__ == '__main__':
    main()
