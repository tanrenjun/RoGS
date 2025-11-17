"""
参数调优脚本：测试不同的分位数参数

该脚本用于寻找最优的 auto_threshold_percentile 参数
目标：达到 min_area = 0.0025m² (0.05m × 0.05m) 的最小分辨率

使用方法:
python run_parameter_tuning.py --percentiles 1 2 3 5 10
"""

import os
import sys
import argparse
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
    """日志记录器"""
    
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
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return Addict(cfg)


def test_single_percentile(
    boundary_points: np.ndarray,
    sem_feat: np.ndarray,
    trajectory_mask: np.ndarray,
    percentile: float,
    min_area: float = 0.0025,
    max_depth: int = 18,
):
    """测试单个分位数参数"""
    
    tree = AdaptiveKDTreeV2(
        bev_resolution=0.05,
        min_area=min_area,
        max_depth=max_depth,
        feature_threshold=None,
        auto_threshold_percentile=percentile,
        geo_weight=0.7,
        sem_weight=0.3,
    )
    
    tree.build(
        boundary_points=boundary_points,
        sem_feat=sem_feat,
        trajectory_mask=trajectory_mask,
        cut_range=5.0
    )
    
    res_stats = tree.get_resolution_stats()
    
    return {
        'percentile': percentile,
        'threshold': tree.feature_threshold,
        'num_leaves': tree.stats['num_leaves'],
        'max_depth': tree.stats['max_depth_reached'],
        'min_resolution': res_stats['min_resolution'],
        'mean_resolution': res_stats['mean_resolution'],
        'median_resolution': res_stats['median_resolution'],
        'build_time': tree.stats['preprocess_time'] + tree.stats['build_time'],
    }


def run_parameter_tuning(
    config_path: str = 'configs/local_nusc_mini.yaml',
    percentiles: list = [1, 2, 3, 5, 10, 15, 20],
    num_images: int = 120,
):
    """运行参数调优"""
    
    # 创建日志
    log_dir = Path("Log")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_parameter_tuning.log"
    
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print("=" * 80)
    print("参数调优：寻找最优分位数")
    print("=" * 80)
    print(f"\n日志文件: {log_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"测试分位数: {percentiles}")
    print(f"目标最小分辨率: 0.05m\n")
    
    try:
        # 加载数据
        print("=" * 80)
        print("步骤 1/3: 加载数据")
        print("=" * 80)
        configs = load_configs(config_path)
        dataset = NuscDataset(configs.dataset, use_label=True, use_depth=False)
        print("  ✓ 数据加载完成\n")
        
        # 语义融合
        print("=" * 80)
        print("步骤 2/3: 语义融合")
        print("=" * 80)
        enriched_pc = fuse_multiview_semantics(
            dataset,
            num_classes=65,
            use_all_cameras=True,
            max_images=num_images,
            verbose=True
        )
        
        xyz_all = enriched_pc["xyz"]
        sem_feat = enriched_pc["sem_feat"]
        boundary_points = xyz_all[:, :2]
        
        # 轨迹掩码
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
        print("  ✓ 预处理完成\n")
        
        # 测试不同分位数
        print("=" * 80)
        print("步骤 3/3: 测试不同分位数参数")
        print("=" * 80)
        
        results = []
        for i, percentile in enumerate(percentiles):
            print(f"\n[{i+1}/{len(percentiles)}] 测试分位数 = {percentile}%")
            print("-" * 60)
            
            result = test_single_percentile(
                boundary_points=boundary_points,
                sem_feat=sem_feat,
                trajectory_mask=trajectory_mask,
                percentile=percentile,
            )
            results.append(result)
            
            print(f"  阈值: {result['threshold']:.6f}")
            print(f"  叶子节点数: {result['num_leaves']:,}")
            print(f"  最大深度: {result['max_depth']}")
            print(f"  分辨率范围: [{result['min_resolution']:.4f}m, -]")
            print(f"  平均分辨率: {result['mean_resolution']:.4f}m")
            print(f"  构建时间: {result['build_time']:.2f}s")
            
            # 检查是否达到目标
            target_res = 0.05
            if result['min_resolution'] <= target_res * 1.05:
                print(f"  ✓ 已达到目标分辨率 (≤{target_res}m)")
            else:
                print(f"  ✗ 未达到目标 ({result['min_resolution']:.4f}m > {target_res}m)")
        
        # 汇总结果
        print("\n" + "=" * 80)
        print("汇总结果")
        print("=" * 80)
        print(f"\n{'分位数':<10} {'阈值':<12} {'叶子数':<12} {'最小分辨率':<12} {'平均分辨率':<12} {'时间(s)':<10}")
        print("-" * 80)
        
        for r in results:
            达标 = "✓" if r['min_resolution'] <= 0.055 else "✗"
            print(f"{r['percentile']:<10.1f} {r['threshold']:<12.6f} {r['num_leaves']:<12,} "
                  f"{r['min_resolution']:<12.4f} {r['mean_resolution']:<12.4f} {r['build_time']:<10.2f}  {达标}")
        
        # 推荐参数
        print("\n" + "=" * 80)
        print("参数推荐")
        print("=" * 80)
        
        达标结果 = [r for r in results if r['min_resolution'] <= 0.055]
        if 达标结果:
            # 选择网格数最少的（效率最高）
            最优 = min(达标结果, key=lambda x: x['num_leaves'])
            print(f"\n推荐分位数: {最优['percentile']}%")
            print(f"  - 叶子节点数: {最优['num_leaves']:,}")
            print(f"  - 最小分辨率: {最优['min_resolution']:.4f}m")
            print(f"  - 平均分辨率: {最优['mean_resolution']:.4f}m")
            print(f"  - 构建时间: {最优['build_time']:.2f}s")
            print(f"\n理由: 在满足分辨率要求的前提下，网格数最少，效率最高")
        else:
            print("\n警告: 所有测试参数均未达到目标分辨率")
            print("建议: 尝试更小的分位数 (如 0.5, 0.1)")
        
        print("\n" + "=" * 80)
        print("调优完成！")
        print("=" * 80)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志文件: {log_file}\n")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='参数调优脚本')
    parser.add_argument('--percentiles', nargs='+', type=float, 
                        default=[1, 2, 3, 5, 10],
                        help='要测试的分位数列表')
    parser.add_argument('--num_images', type=int, default=120,
                        help='使用的图像数量')
    
    args = parser.parse_args()
    
    run_parameter_tuning(
        percentiles=args.percentiles,
        num_images=args.num_images,
    )
