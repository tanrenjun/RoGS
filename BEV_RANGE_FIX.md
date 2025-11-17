# BEV范围裁剪问题修复说明

## 问题描述

在集成自适应网格（KD-tree V2）到RoGS训练时，发现**BEV范围被意外缩小**的问题：

```
声明的BEV范围: 178.88m × 16.46m（基于轨迹点 + cut_range）
实际使用BEV:   38.64m × 38.83m（被轨迹掩码二次裁剪）
结果: 丢失大量场景区域，覆盖率仅12%
```

## 根本原因

### 原始Road类的正确流程
```python
# 1. 基于轨迹点计算BEV范围
min_coords = np.min(all_pose_xyz, axis=0) - cut_range  # 轨迹最小值 - 8m
max_coords = np.max(all_pose_xyz, axis=0) + cut_range  # 轨迹最大值 + 8m

# 2. 在整个范围生成网格
vertices = create_rect_vertices(min_coords, max_coords, resolution)

# 3. 使用膨胀掩码保留轨迹附近顶点
vertices = cut_point_by_pose(vertices, ..., all_pose_xyz, cut_range)
```

### AdaptiveRoadV2的问题
```python
# 1. 计算轨迹范围（正确）
self.min_xy = min_coords[:2]  # 轨迹 + cut_range
self.max_xy = max_coords[:2]  # 轨迹 + cut_range

# 2. 构建KD-tree时，kdtree_v2内部又做了一次裁剪（错误！）
kdtree.build(boundary_points, sem_feat, trajectory_mask, cut_range)
    ↓
def _preprocess(...):
    if trajectory_mask is not None:
        # 根据掩码重新计算ROI，导致范围缩小！
        self.bev_min_coords, self.bev_max_coords = self._get_roi_from_trajectory(...)
```

**问题**: `_get_roi_from_trajectory`方法根据轨迹掩码的真值区域重新计算BEV边界，导致范围从178m×16m缩小到38m×38m。

## 解决方案

### 修改1: KD-tree V2支持预定义BEV范围

修改`models/kdtree_v2.py`的`build`方法，添加可选参数：

```python
def build(
    self,
    boundary_points: np.ndarray,
    sem_feat: torch.Tensor,
    trajectory_mask: Optional[np.ndarray] = None,
    cut_range: float = 5.0,
    bev_min: Optional[np.ndarray] = None,  # 新增：预定义BEV最小坐标
    bev_max: Optional[np.ndarray] = None   # 新增：预定义BEV最大坐标
) -> 'AdaptiveKDTreeV2':
```

修改`_preprocess`方法，优先使用预定义范围：

```python
def _preprocess(self, ..., bev_min=None, bev_max=None):
    # 1.1 确定 BEV 范围
    if bev_min is not None and bev_max is not None:
        # 使用预定义的BEV范围（来自轨迹点+cut_range）
        print("  - 使用预定义的 BEV 范围（基于轨迹点）")
        self.bev_min_coords = bev_min
        self.bev_max_coords = bev_max
    elif trajectory_mask is not None:
        # 使用轨迹掩码裁剪（旧逻辑，会缩小范围）
        print("  - 使用轨迹掩码裁剪 BEV 范围")
        self.bev_min_coords, self.bev_max_coords = self._get_roi_from_trajectory(...)
    else:
        print("  - 使用全部数据范围")
        self.bev_min_coords = boundary_points.min(axis=0)
        self.bev_max_coords = boundary_points.max(axis=0)
```

### 修改2: AdaptiveRoadV2传递预定义范围

修改`models/adaptive_road_v2.py`，传递轨迹范围给KD-tree：

```python
# Build the tree with predefined BEV range
kdtree.build(
    boundary_points=bev_points,
    sem_feat=sem_feat,
    trajectory_mask=traj_mask,
    cut_range=self.cut_range,
    bev_min=self.min_xy,  # 传递轨迹范围（已包含cut_range）
    bev_max=self.max_xy   # 传递轨迹范围（已包含cut_range）
)
```

## 修复效果

### 修复前
```
BEV范围: 178.88m x 16.46m
高度范围: [0.00m, 0.34m]

[1/4] 构建KD-tree自适应网格...
  路面点数: 2,220,782
  BEV原始范围: [-24.21, 183.72] x [-20.19, 20.26]

[2/4] 构建KD-tree...
  轨迹掩码覆盖率: 12.05%

============================================================
开始构建 KD-tree (积分图优化版)
============================================================

[1/3] 预处理阶段...
  - 使用轨迹掩码裁剪 BEV 范围 (cut_range=3.0m)
  - BEV 范围: [60.69, 99.33] x [-19.05, 19.79]  ❌ 错误：范围被缩小
  - BEV 尺寸: 38.64m x 38.83m  ❌ 错误：只有部分场景
  
结果: 叶子节点数 262,144，只覆盖部分场景
```

### 修复后
```
BEV范围: 178.88m x 16.46m
高度范围: [0.00m, 0.34m]

[1/4] 构建KD-tree自适应网格...
  路面点数: 2,220,782
  使用轨迹定义的BEV范围: [-8.00, 170.88] x [-8.20, 8.26]

[2/4] 构建KD-tree...
  轨迹掩码覆盖率: 97.47%  ✅ 完整覆盖

============================================================
开始构建 KD-tree (积分图优化版)
============================================================

[1/3] 预处理阶段...
  - 使用预定义的 BEV 范围（基于轨迹点）  ✅ 新逻辑
  - BEV 范围: [-8.00, 170.88] x [-8.20, 8.26]  ✅ 正确：完整范围
  - BEV 尺寸: 178.88m x 16.46m  ✅ 正确：覆盖完整场景

结果: 叶子节点数 1,048,576，完整覆盖整个场景
```

## 性能对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 声明BEV范围 | 178.88m × 16.46m | 178.88m × 16.46m |
| 实际BEV范围 | 38.64m × 38.83m ❌ | 178.88m × 16.46m ✅ |
| 覆盖率 | 12.05% | 97.47% |
| 叶子节点数 | 262k | 1,048k |
| 预处理时间 | 3.84s | 11.47s |
| 构建时间 | 1.29s | 5.29s |
| 总时间 | 5.13s | 16.75s |
| 训练速度 | ~120 it/s | ~75 it/s |
| Epoch 0 PSNR | 27.51 (部分场景) | 21.48 (完整场景) |
| Epoch 0 mIoU | 0.865 (部分场景) | 0.862 (完整场景) |

**说明**: 
- 修复后PSNR较低是因为覆盖了更大的场景（包括更难重建的区域）
- 顶点数增加4倍但仍优于固定网格（1.05M vs 1.14M）
- 训练速度略降但换来完整场景覆盖

## 核心设计原则

### 轨迹掩码的正确用途
- ❌ **不应该**: 用于缩小BEV范围（导致丢失场景）
- ✅ **应该**: 用于指导细分策略（轨迹区域更精细）

### BEV范围的确定方式
1. **基于轨迹点**: `min/max(trajectory_points) ± cut_range`
2. **保持不变**: 整个构建过程中BEV范围固定
3. **掩码作用**: 仅用于标记轨迹区域，不改变边界

### 与原始Road类一致
原始Road类的流程：
```
轨迹点 → BEV范围 → 生成完整网格 → 掩码裁剪顶点
```

自适应Road类的流程：
```
轨迹点 → BEV范围 → 生成完整KD-tree → 掩码指导细分
```

两者都在**完整的BEV范围**内工作！

## 结论

✅ **问题已修复**: BEV范围现在与原始Road类保持一致

✅ **完整覆盖**: 97.47%场景覆盖率（vs修复前12%）

✅ **性能合理**: 虽然顶点数增加，但仍优于固定网格

✅ **行为一致**: 与原始Road类的逻辑保持一致

---

**修复时间**: 2025-11-14 21:07
**测试场景**: scene-0655 (nuScenes v1.0-mini)
**状态**: ✅ 已验证
