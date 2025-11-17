# 自适应网格 KD-tree (V2) - 积分图优化版

## 核心改进

1. **预计算 BEV 特征图**：几何特征 + 语义特征解耦
2. **积分图（DP）加速**：O(1) 区域查询，30x+ 性能提升
3. **轨迹裁剪**：减少 77% 无效计算区域
4. **可调分位数阈值**：灵活控制细分程度

## 文件说明

### 核心代码
- `models/kdtree_v2.py`: 积分图优化的 KD-tree 实现
- `utils/trajectory_utils.py`: 车辆轨迹提取和掩码工具

### 运行脚本
- `run_grid_comparison.py`: 固定网格 vs 自适应网格完整对比
- `run_parameter_tuning.py`: 参数调优脚本，寻找最优分位数

### 日志目录
- `Log/`: 所有实验日志按时间和任务自动保存

## 快速开始

### 1. 运行完整对比流程

```bash
conda activate rogs
python run_grid_comparison.py
```

**输出**：
- 日志文件：`Log/YYYYMMDD_HHMMSS_grid_comparison.log`
- 对比固定网格和自适应网格的性能、网格数、分辨率等

### 2. 参数调优

寻找最优的分位数参数（达到 0.05m 最小分辨率）：

```bash
# 测试默认分位数 [1, 2, 3, 5, 10]
python run_parameter_tuning.py

# 自定义分位数
python run_parameter_tuning.py --percentiles 0.5 1 2 3
```

**输出**：
- 日志文件：`Log/YYYYMMDD_HHMMSS_parameter_tuning.log`
- 汇总表格和参数推荐

## 关键参数说明

### `AdaptiveKDTreeV2` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bev_resolution` | 0.05 | BEV 特征图分辨率 (m/pixel) |
| `min_area` | 0.0025 | 最小区域面积 (m²) = 0.05×0.05 |
| `max_depth` | 18 | 最大树深度 |
| `feature_threshold` | None | 特征阈值，None=自动 |
| `auto_threshold_percentile` | 10.0 | 自动阈值的分位数 (0-100) |
| `geo_weight` | 0.7 | 几何特征权重 |
| `sem_weight` | 0.3 | 语义特征权重 |

### 分位数调优建议

- **分位数越小，细分越多**
  - 1% → 更激进的细分，更接近 min_area
  - 20% → 更保守，更少的网格
  
- **实验结果参考**：
  - 5%: 最小分辨率 0.0395m，589k 叶子节点
  - 10%: 最小分辨率 ~0.04m，适中
  - 20%: 最小分辨率 >0.04m，较少细分

## 性能对比（scene-0655, 120 images）

| 方案 | 网格数 | 顶点数 | 分辨率 | 构建时间 | 存储 |
|------|--------|--------|--------|----------|------|
| 固定网格 (0.05m) | 3.36M | 3.37M | 0.05m | 0.04s | 25.71MB |
| 自适应网格 (5%) | 589k | 2.36M | 0.04-5.06m | 10.51s | 17.99MB |

**优势**：
- 网格减少：82.48%
- 顶点减少：30.03%
- 存储减少：30.0%
- 动态分辨率：复杂区域 0.04m，简单区域 5m+

## 使用示例

```python
from models.kdtree_v2 import AdaptiveKDTreeV2

# 创建 KD-tree
tree = AdaptiveKDTreeV2(
    bev_resolution=0.05,
    min_area=0.0025,
    max_depth=18,
    auto_threshold_percentile=5.0,  # 5分位数
)

# 构建树
tree.build(
    boundary_points=boundary_points,  # (N, 2)
    sem_feat=sem_feat,                # (N, C)
    trajectory_mask=trajectory_mask,  # (H, W) bool
    cut_range=5.0,                    # 轨迹裁剪范围
)

# 获取顶点
vertices = tree.get_vertices()  # (M, 2) torch.Tensor

# 获取统计
stats = tree.get_resolution_stats()
print(f"分辨率范围: [{stats['min_resolution']:.4f}, {stats['max_resolution']:.4f}]m")
```

## 日志示例

所有日志自动保存到 `Log/` 目录，格式：`YYYYMMDD_HHMMSS_任务名.log`

例如：
- `20251114_184215_grid_comparison.log`: 对比实验
- `20251114_185630_parameter_tuning.log`: 参数调优

## 注意事项

1. **分位数选择**：
   - 如果最小分辨率 > 目标值，降低分位数
   - 如果网格太多（>100万），提高分位数

2. **轨迹掩码**：
   - 始终使用轨迹掩码，减少 77% 计算量
   - `cut_range=5.0m` 通常足够

3. **深度限制**：
   - `max_depth=18` 对应最小 0.05m (208m场景)
   - 更大场景需要更大深度

## 性能优化技术

1. **积分图（Summed Area Table）**：
   - 预计算：O(H×W) 一次性
   - 查询：O(1) 常数时间
   - 原理：`sum(rect) = dp[y2,x2] - dp[y1,x2] - dp[y2,x1] + dp[y1,y1]`

2. **特征计算解耦**：
   - 阶段1：构建 BEV 特征图（7s）
   - 阶段2：树递归查询（3s）
   - 避免重复计算特征

3. **轨迹裁剪**：
   - 车辆轨迹 ±5m 区域
   - 减少 BEV 面积 77%
   - 仅在有意义区域细分

## 后续改进方向

1. **并行化**：树构建可以并行化
2. **GPU 加速**：BEV 特征图构建可用 GPU
3. **动态阈值**：不同区域使用不同阈值
4. **多尺度融合**：结合多个分辨率的 BEV
