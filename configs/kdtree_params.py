"""
参数对比实验配置

本文件定义了不同的 KD-tree 参数配置方案，用于对比实验。
"""

# KD-tree 参数配置方案
KDTREE_CONFIGS = {
    # 方案1: 极细网格 - 最大细分
    "ultra_fine": {
        "min_area": 0.1,           # 最小区域 0.1 m²
        "max_depth": 12,           # 最大深度 12 层
        "feature_threshold": 0.02, # 低阈值，易触发细分
        "min_points_per_region": 20,
        "description": "极细网格划分，适合语义复杂场景"
    },
    
    # 方案2: 细网格 - 高精度
    "fine": {
        "min_area": 0.5,           # 最小区域 0.5 m²
        "max_depth": 10,           # 最大深度 10 层
        "feature_threshold": 0.05, # 中低阈值
        "min_points_per_region": 50,
        "description": "细网格划分，平衡精度和效率"
    },
    
    # 方案3: 中等网格 - 平衡（推荐）
    "medium": {
        "min_area": 1.0,           # 最小区域 1 m²
        "max_depth": 8,            # 最大深度 8 层
        "feature_threshold": 0.1,  # 中等阈值
        "min_points_per_region": 100,
        "description": "中等网格，推荐方案，平衡各方面性能"
    },
    
    # 方案4: 粗网格 - 高效率
    "coarse": {
        "min_area": 2.0,           # 最小区域 2 m²
        "max_depth": 6,            # 最大深度 6 层
        "feature_threshold": 0.15, # 较高阈值
        "min_points_per_region": 200,
        "description": "粗网格划分，快速计算"
    },
    
    # 方案5: 极粗网格 - 最快速度
    "ultra_coarse": {
        "min_area": 5.0,           # 最小区域 5 m²
        "max_depth": 5,            # 最大深度 5 层
        "feature_threshold": 0.2,  # 高阈值
        "min_points_per_region": 500,
        "description": "极粗网格，接近固定网格"
    },
    
    # 方案6: 原始论文配置（参考 demo）
    "demo_default": {
        "min_area": 0.5,
        "max_depth": 7,
        "feature_threshold": 0.05,
        "min_points_per_region": 50,
        "description": "Demo 默认配置"
    },
}


# 固定网格参数配置（用于对比）
FIXED_GRID_CONFIGS = {
    # 对应 ROGS 原始配置
    "rogs_original": {
        "resolution": 0.05,  # 5cm 分辨率
        "description": "ROGS 原始固定网格（0.05m）"
    },
    
    # 更粗的固定网格
    "coarse_10cm": {
        "resolution": 0.1,   # 10cm 分辨率
        "description": "粗固定网格（0.1m）"
    },
    
    # 中等固定网格
    "medium_20cm": {
        "resolution": 0.2,   # 20cm 分辨率
        "description": "中等固定网格（0.2m）"
    },
    
    # 粗固定网格
    "coarse_50cm": {
        "resolution": 0.5,   # 50cm 分辨率
        "description": "粗固定网格（0.5m）"
    },
}


def get_kdtree_config(name="medium"):
    """获取 KD-tree 配置"""
    if name not in KDTREE_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(KDTREE_CONFIGS.keys())}")
    return KDTREE_CONFIGS[name].copy()


def get_fixed_grid_config(name="rogs_original"):
    """获取固定网格配置"""
    if name not in FIXED_GRID_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(FIXED_GRID_CONFIGS.keys())}")
    return FIXED_GRID_CONFIGS[name].copy()


def list_all_configs():
    """列出所有可用配置"""
    print("=" * 80)
    print("KD-tree 自适应网格配置:")
    print("=" * 80)
    for name, cfg in KDTREE_CONFIGS.items():
        print(f"\n[{name}] {cfg['description']}")
        print(f"  min_area: {cfg['min_area']} m²")
        print(f"  max_depth: {cfg['max_depth']}")
        print(f"  feature_threshold: {cfg['feature_threshold']}")
        print(f"  min_points_per_region: {cfg['min_points_per_region']}")
    
    print("\n" + "=" * 80)
    print("固定网格配置:")
    print("=" * 80)
    for name, cfg in FIXED_GRID_CONFIGS.items():
        print(f"\n[{name}] {cfg['description']}")
        print(f"  resolution: {cfg['resolution']} m")


# 参数影响说明
PARAMETER_GUIDE = """
KD-tree 参数调整指南
==================

1. min_area (最小区域面积，单位: m²)
   作用: 控制叶子节点的最小面积，阻止过度细分
   - 越小 → 网格越细，顶点越多，计算量越大，精度越高
   - 越大 → 网格越粗，顶点越少，计算速度越快
   推荐范围: 0.1 ~ 5.0 m²
   
2. max_depth (最大树深度)
   作用: 限制 KD-tree 的最大层数
   - 越大 → 允许更深的细分，但可能过拟合
   - 越小 → 限制细分程度，保持粗粒度
   推荐范围: 5 ~ 12 层
   计算公式: 深度 d 可细分为 2^d 个区域
   
3. feature_threshold (语义复杂度阈值)
   作用: 决定区域是否需要细分的语义判据
   - 越小 → 更容易触发细分，对语义变化敏感
   - 越大 → 不易细分，只在语义差异大时细分
   推荐范围: 0.02 ~ 0.2
   
4. min_points_per_region (区域最少点数)
   作用: 要求每个区域至少包含的点数
   - 越小 → 允许小区域存在
   - 越大 → 强制区域包含足够的点
   推荐范围: 20 ~ 500

参数组合建议
============

场景1: 城市道路（语义复杂）
- min_area: 0.5
- max_depth: 10
- feature_threshold: 0.05
- 特点: 细网格捕捉复杂语义

场景2: 高速公路（语义简单）
- min_area: 2.0
- max_depth: 6
- feature_threshold: 0.15
- 特点: 粗网格提高效率

场景3: 停车场（中等复杂度）
- min_area: 1.0
- max_depth: 8
- feature_threshold: 0.1
- 特点: 平衡精度和速度
"""


if __name__ == "__main__":
    list_all_configs()
    print("\n" + PARAMETER_GUIDE)
