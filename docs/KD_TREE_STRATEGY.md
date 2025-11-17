# KD-tree 自适应网格划分策略 - 改进方案

## 目标

实现真正的自适应网格划分：
- **复杂区域**（路口、车道线密集）：细分到 0.05m 或更细
- **简单区域**（直道、空旷处）：保持粗分辨率（0.5-2.0m）
- **效率**：通过车辆轨迹预处理减少计算量

## 当前策略分析

### 现有细分条件（过于复杂）

```python
# 条件 1: 高语义复杂度 + 至少 5 个点
if semantic_complexity > threshold and point_count >= 5:
    subdivide()

# 条件 2: 高点密度（> 20 点/m²）+ 至少 10 个点
if point_density > 20 and point_count >= 10:
    subdivide()

# 条件 3: 中等密度 + 区域很大 + 至少 10 个点
if point_density > 5 and area > min_area * 4 and point_count >= 10:
    subdivide()

# 条件 4: 区域大于 2 倍 min_area + 至少 3 个点
if area > min_area * 2 and point_count >= 3:
    subdivide()
```

### 问题

1. **条件过多**：4 个条件相互耦合，难以理解和调试
2. **硬编码阈值**：点密度 20/5、点数 10/5/3 等都是经验值
3. **效率问题**：每次判断都要计算点数、密度、语义复杂度
4. **语义复杂度计算复杂**：熵 + 方差 + 置信度惩罚，计算开销大

---

## 你的建议（简化策略）✨

### 核心思想

```
对于每个 KD-tree 节点（矩形区域）：
    计算该区域的"特征强度"（几何 + 语义）
    如果 特征强度 > threshold:
        继续细分
    否则:
        停止，作为叶子节点
```

### 优势

1. **简单清晰**：只有一个判断条件
2. **可解释性强**：特征强度直接反映区域复杂度
3. **易于调试**：只需调整一个阈值参数
4. **BEV 视角统一**：几何和语义在同一 BEV 空间计算

---

## 改进方案：BEV 特征强度驱动

### 1. 特征强度定义

#### 方案 A：几何梯度 + 语义熵（推荐）

```python
def calculate_feature_intensity(region_points, region_semantics):
    """
    计算 BEV 区域的特征强度
    
    Args:
        region_points: (N, 2) BEV 平面点坐标
        region_semantics: (N, C) 语义特征
    
    Returns:
        intensity: 特征强度值 [0, 1]
    """
    # 1. 几何复杂度：点云密度变化（梯度）
    point_density = len(region_points) / region_area
    
    # 在 BEV 上计算空间分布的方差
    spatial_variance = np.var(region_points, axis=0).mean()
    geometric_intensity = min(spatial_variance / 1.0, 1.0)  # 归一化
    
    # 2. 语义复杂度：类别熵
    mean_semantic = region_semantics.mean(axis=0)
    mean_semantic /= (mean_semantic.sum() + 1e-8)
    semantic_entropy = -np.sum(mean_semantic * np.log(mean_semantic + 1e-8))
    semantic_entropy /= np.log(num_classes)  # 归一化到 [0, 1]
    
    # 3. 综合特征强度
    intensity = 0.5 * geometric_intensity + 0.5 * semantic_entropy
    
    return intensity
```

#### 方案 B：仅基于点密度（最简单）

```python
def calculate_feature_intensity(region_points, region_area):
    """基于点密度的特征强度"""
    point_density = len(region_points) / region_area
    
    # 密度越高，特征强度越高
    # 使用对数缩放，避免极端值
    intensity = np.log(1 + point_density) / np.log(1 + 100)  # 假设最大密度 100
    
    return min(intensity, 1.0)
```

#### 方案 C：BEV 图像梯度（最直观）

```python
def calculate_feature_intensity_from_bev(bev_region):
    """
    从 BEV 图像块计算特征强度
    
    Args:
        bev_region: (H, W, C) BEV 图像区域（可以是语义图或 RGB）
    
    Returns:
        intensity: 特征强度
    """
    # 计算图像梯度（Sobel 或其他）
    gray = cv2.cvtColor(bev_region, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅值
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    intensity = gradient_magnitude.mean() / 255.0  # 归一化
    
    return intensity
```

### 2. 简化的细分逻辑

```python
def _should_subdivide_by_features(self, node, feature_stats):
    """
    简化的细分判断：仅基于特征强度
    """
    # 计算区域特征强度
    intensity = self._calculate_feature_intensity(node)
    
    # 唯一判断条件
    if intensity > self.feature_threshold:
        return True
    
    return False
```

---

## 车辆轨迹预处理（加速策略）✨

你的建议非常好！参考原始 ROGS 的做法。

### 原始 ROGS 的预处理

<function_calls>
<invoke name="grep_search">
<parameter name="isRegexp">false