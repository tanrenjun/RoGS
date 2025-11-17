"""
自适应 KD-tree 参数配置 - 实现真正的多分辨率自适应
目标：复杂区域细分（< 0.05m），简单区域稀疏（> 0.2m）
"""

import numpy as np


def get_adaptive_kdtree_config(preset="balanced"):
    """
    获取真正自适应的 KD-tree 配置
    
    核心思想：
    1. min_area 要非常小（< 0.0025 m²，对应 0.05m×0.05m），允许细分
    2. max_depth 要足够大（10-15），支持深度细分
    3. feature_threshold 要合理，控制何时停止细分
    
    Args:
        preset: 预设名称
            - "ultra_fine": 极细分辨率，复杂场景
            - "fine": 细分辨率，城市道路
            - "balanced": 平衡配置（推荐）
            - "coarse": 粗分辨率，高速公路
    
    Returns:
        dict: KD-tree 配置参数
    """
    
    configs = {
        # 极细分辨率：最小网格 0.025m，适合路口、停车场等极复杂场景
        "ultra_fine": {
            "min_area": 0.000625,      # 0.025m × 0.025m = 0.000625 m²
            "max_depth": 20,           # 深度 20，支持极细分辨率
            "feature_threshold": 0.02, # 低阈值，易触发细分
            "min_points_per_region": 10,  # 降低点数要求
            "description": "最细分辨率 0.025m，适合极复杂场景（路口、停车场）"
        },
        
        # 细分辨率：最小网格 0.05m（与原始相同），适合城市道路
        "fine": {
            "min_area": 0.0025,        # 0.05m × 0.05m = 0.0025 m²
            "max_depth": 18,           # 深度 18，实现 0.05m 分辨率
            "feature_threshold": 0.05, # 中低阈值
            "min_points_per_region": 20,
            "description": "最细分辨率 0.05m（原始精度），适合城市道路"
        },
        
        # 平衡配置：最小网格 0.10m，平衡精度和效率
        "balanced": {
            "min_area": 0.01,          # 0.10m × 0.10m = 0.01 m²
            "max_depth": 16,           # 深度 16
            "feature_threshold": 0.08, # 中等阈值
            "min_points_per_region": 30,
            "description": "最细分辨率 0.10m，平衡精度和效率"
        },
        
        # 粗分辨率：最小网格 0.20m，适合高速公路
        "coarse": {
            "min_area": 0.04,          # 0.20m × 0.20m = 0.04 m²
            "max_depth": 14,           # 深度 14
            "feature_threshold": 0.12, # 高阈值，不易细分
            "min_points_per_region": 50,
            "description": "最细分辨率 0.20m，适合简单高速路"
        }
    }
    
    if preset not in configs:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(configs.keys())}")
    
    return configs[preset]


def calculate_resolution_range(min_area, max_depth):
    """
    计算 KD-tree 的分辨率范围
    
    Args:
        min_area: 最小区域面积（m²）
        max_depth: 最大树深度
    
    Returns:
        dict: 包含最小和最大分辨率信息
    """
    # 假设正方形区域
    min_resolution = np.sqrt(min_area)  # 最细分辨率（m）
    
    # 最大区域取决于初始边界和深度
    # 假设初始区域为 100m × 100m
    initial_area = 100.0 * 100.0
    max_area = initial_area / (2 ** max_depth)
    max_resolution = np.sqrt(max_area)  # 最粗分辨率（m）
    
    return {
        "min_resolution_m": min_resolution,
        "max_resolution_m": max_resolution,
        "min_area_m2": min_area,
        "max_area_m2": max_area,
        "resolution_ratio": max_resolution / min_resolution,
        "max_depth": max_depth
    }


def print_all_adaptive_configs():
    """打印所有自适应配置及其分辨率范围"""
    print("=" * 80)
    print("自适应 KD-tree 参数配置（真正的多分辨率自适应）")
    print("=" * 80)
    print()
    
    presets = ["ultra_fine", "fine", "balanced", "coarse"]
    
    print("目标：在复杂区域细分（< 0.05m），在简单区域稀疏（> 0.2m）")
    print("对比：原始固定网格为 0.05m")
    print()
    print("-" * 80)
    
    for preset in presets:
        config = get_adaptive_kdtree_config(preset)
        res_info = calculate_resolution_range(config["min_area"], config["max_depth"])
        
        print(f"\n[{preset.upper()}]")
        print(f"  描述: {config['description']}")
        print(f"  参数:")
        print(f"    - min_area: {config['min_area']:.6f} m²")
        print(f"    - max_depth: {config['max_depth']}")
        print(f"    - feature_threshold: {config['feature_threshold']:.3f}")
        print(f"    - min_points_per_region: {config['min_points_per_region']}")
        print(f"  分辨率范围:")
        print(f"    - 最细: {res_info['min_resolution_m']:.4f} m/pixel")
        print(f"    - 最粗: {res_info['max_resolution_m']:.4f} m/pixel")
        print(f"    - 比值: {res_info['resolution_ratio']:.1f}x")
        
        # 与原始 0.05m 对比
        comparison = "细于" if res_info['min_resolution_m'] < 0.05 else \
                     "等于" if abs(res_info['min_resolution_m'] - 0.05) < 0.001 else "粗于"
        print(f"    - 与原始 0.05m 对比: {comparison}原始（{res_info['min_resolution_m']/0.05:.2f}x）")
    
    print()
    print("-" * 80)
    print("\n推荐使用:")
    print("  - 城市复杂场景: 'fine' （最细 0.05m，与原始相同）")
    print("  - 一般场景: 'balanced' （最细 0.10m，2倍于原始）")
    print("  - 如果需要超越原始精度: 'ultra_fine' （最细 0.025m，0.5倍原始）")
    print()
    print("=" * 80)


def get_comparison_configs():
    """获取用于对比实验的配置组"""
    return {
        # 固定网格基准
        "fixed_0.05m": {"type": "fixed", "resolution": 0.05},
        "fixed_0.10m": {"type": "fixed", "resolution": 0.10},
        "fixed_0.20m": {"type": "fixed", "resolution": 0.20},
        
        # 自适应 KD-tree
        "kdtree_ultra_fine": {"type": "kdtree", **get_adaptive_kdtree_config("ultra_fine")},
        "kdtree_fine": {"type": "kdtree", **get_adaptive_kdtree_config("fine")},
        "kdtree_balanced": {"type": "kdtree", **get_adaptive_kdtree_config("balanced")},
        "kdtree_coarse": {"type": "kdtree", **get_adaptive_kdtree_config("coarse")},
    }


if __name__ == "__main__":
    print_all_adaptive_configs()
    
    print("\n\n使用示例:")
    print("-" * 80)
    print("""
# 方式1: 使用预设配置（推荐）
from configs.kdtree_adaptive_params import get_adaptive_kdtree_config

config = get_adaptive_kdtree_config("fine")  # 最细 0.05m
vertices, tree = build_kdtree_on_bev(xyz, sem_feat, **config)

# 方式2: 自定义配置
vertices, tree = build_kdtree_on_bev(
    xyz, sem_feat,
    min_area=0.0025,        # 0.05m × 0.05m，与原始相同
    max_depth=12,           # 支持深度细分
    feature_threshold=0.05  # 控制细分程度
)
    """)
