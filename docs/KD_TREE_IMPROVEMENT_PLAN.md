# KD-tree 改进方案：简化策略 + 轨迹预处理

## 一、当前问题

### 1. 细分策略过于复杂
```python
# 当前有 4 个细分条件，相互耦合
- 条件 1: 语义复杂度 > threshold AND 点数 >= 5
- 条件 2: 点密度 > 20 AND 点数 >= 10  
- 条件 3: 点密度 > 5 AND 面积 > 4×min_area AND 点数 >= 10
- 条件 4: 面积 > 2×min_area AND 点数 >= 3
```

**问题**：
- 条件太多，难以理解和调试
- 硬编码阈值（20, 5, 10, 3）缺乏理论依据
- 每次判断都要计算多个指标，效率低

### 2. 语义复杂度计算开销大
```python
semantic_complexity = 0.5 * entropy + 0.3 * variance + 0.2 * confidence_penalty
```
- 需要遍历区域内所有点
- 计算熵、方差、置信度
- 对于 220 万个点，计算量巨大

### 3. 缺少预处理优化
- 对整个 BEV 区域（208m × 40m）进行细分
- 包含大量车辆未到达的区域
- 浪费计算资源

---

## 二、改进方案

### 方案 A：简化细分策略（你的建议）✨

#### 核心思想
```
对于每个 KD-tree 节点（矩形区域）：
    特征强度 = 计算_BEV_特征强度(区域)
    if 特征强度 > threshold:
        继续细分
    else:
        停止，作为叶子节点
```

#### 实现代码

**选项 1：基于点密度（最简单，推荐）**

```python
def _calculate_feature_intensity(self, node: KDTreeNode) -> float:
    """
    计算区域的特征强度（基于点密度）
    
    Args:
        node: KD-tree 节点
        
    Returns:
        intensity: 特征强度 [0, 1]
    """
    # 获取区域内的点
    region_points = self._get_points_in_region(node)
    point_count = len(region_points)
    
    if point_count == 0:
        return 0.0
    
    # 计算点密度（点/m²）
    point_density = point_count / max(node.area, 1e-6)
    
    # 密度归一化到 [0, 1]
    # 假设：密度 > 100 点/m² 为高复杂度
    intensity = min(point_density / 100.0, 1.0)
    
    return intensity


def _should_subdivide_by_features(self, node: KDTreeNode, feature_stats: dict) -> bool:
    """
    简化的细分判断：仅基于特征强度
    """
    intensity = self._calculate_feature_intensity(node)
    
    # 唯一判断条件
    return intensity > self.feature_threshold
```

**选项 2：几何 + 语义混合**

```python
def _calculate_feature_intensity(self, node: KDTreeNode) -> float:
    """
    计算区域的特征强度（几何 + 语义）
    
    Returns:
        intensity: 特征强度 [0, 1]
    """
    region_points = self._get_points_in_region(node)
    point_count = len(region_points)
    
    if point_count == 0:
        return 0.0
    
    # 1. 几何强度：点密度
    point_density = point_count / max(node.area, 1e-6)
    geometric_intensity = min(point_density / 100.0, 1.0)
    
    # 2. 语义强度：类别熵（简化版）
    if self.feature_map is not None:
        mask = np.all((self.boundary_points >= node.min_coords) & 
                     (self.boundary_points <= node.max_coords), axis=1)
        region_features = self.feature_map[mask]
        
        if len(region_features) > 0:
            mean_features = np.mean(region_features, axis=0)
            mean_features /= (np.sum(mean_features) + 1e-8)
            entropy = -np.sum(mean_features * np.log(mean_features + 1e-8))
            max_entropy = np.log(len(mean_features))  # 最大熵
            semantic_intensity = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            semantic_intensity = 0.0
    else:
        semantic_intensity = 0.0
    
    # 3. 综合强度（可调权重）
    intensity = 0.7 * geometric_intensity + 0.3 * semantic_intensity
    
    return intensity
```

**选项 3：BEV 图像梯度（最直观）**

```python
def _calculate_feature_intensity_from_bev(self, node: KDTreeNode, bev_image: np.ndarray) -> float:
    """
    从 BEV 图像计算区域的特征强度
    
    Args:
        node: KD-tree 节点
        bev_image: BEV 语义图或 RGB 图 (H, W, C)
        
    Returns:
        intensity: 特征强度 [0, 1]
    """
    # 将节点坐标转换为图像像素坐标
    min_xy = node.min_coords
    max_xy = node.max_coords
    
    # 提取 BEV 图像中对应的区域
    # （需要根据 BEV 图像的分辨率和范围进行坐标转换）
    # 这里假设已有转换函数
    bev_region = extract_bev_region(bev_image, min_xy, max_xy)
    
    if bev_region.size == 0:
        return 0.0
    
    # 计算图像梯度
    if len(bev_region.shape) == 3:
        gray = cv2.cvtColor(bev_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = bev_region
    
    # Sobel 梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅值
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    intensity = gradient_magnitude.mean() / 255.0  # 归一化
    
    return intensity
```

#### 参数调整

```python
# configs/kdtree_adaptive_params.py
configs = {
    "fine": {
        "min_area": 0.0025,        # 0.05m × 0.05m
        "max_depth": 18,
        "feature_threshold": 0.2,  # 特征强度阈值（0-1）
        # 移除 min_points_per_region，不再需要
    }
}
```

---

### 方案 B：车辆轨迹预处理（加速策略）✨

#### 核心思想

参考 ROGS 原始方法，**只对车辆轨迹附近区域进行 KD-tree 细分**。

#### 实现步骤

**Step 1：提取车辆轨迹点**

```python
def get_vehicle_trajectory(dataset):
    """
    从数据集提取车辆轨迹
    
    Returns:
        poses_xy: (N, 2) 车辆在 BEV 平面的位置
    """
    poses_xy = []
    
    for idx in range(len(dataset)):
        # 获取相机位姿（即车辆位姿）
        camera2world = dataset.camera2world_all[idx]  # (4, 4)
        pose_xy = camera2world[:2, 3]  # 提取 x, y 坐标
        poses_xy.append(pose_xy)
    
    poses_xy = np.array(poses_xy)
    return poses_xy
```

**Step 2：创建轨迹掩码（参考 `cut_point_by_pose`）**

```python
def create_trajectory_mask(poses_xy, min_coords, max_coords, resolution=0.05, cut_range=5.0):
    """
    创建车辆轨迹附近的掩码
    
    Args:
        poses_xy: (N, 2) 车辆轨迹点
        min_coords: BEV 最小坐标 [x_min, y_min]
        max_coords: BEV 最大坐标 [x_max, y_max]
        resolution: BEV 分辨率（米/像素）
        cut_range: 轨迹周围的范围（米）
        
    Returns:
        mask: (H, W) 布尔掩码，True 表示需要细分的区域
    """
    # 计算 BEV 图像尺寸
    bev_size_x = int((max_coords[0] - min_coords[0]) / resolution)
    bev_size_y = int((max_coords[1] - min_coords[1]) / resolution)
    
    # 将轨迹点转换为像素坐标
    pixel_xy = np.zeros_like(poses_xy)
    pixel_xy[:, 0] = (poses_xy[:, 0] - min_coords[0]) / resolution
    pixel_xy[:, 1] = (poses_xy[:, 1] - min_coords[1]) / resolution
    pixel_xy = np.unique(pixel_xy.round(), axis=0)
    
    # 裁剪到有效范围
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, bev_size_x - 1)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, bev_size_y - 1)
    pixel_xy = pixel_xy.astype(np.int32)
    
    # 创建掩码
    mask = np.zeros((bev_size_x, bev_size_y), dtype=np.uint8)
    mask[pixel_xy[:, 0], pixel_xy[:, 1]] = 1
    
    # 膨胀掩码（扩展到轨迹周围 cut_range 米）
    kernel_size = int(cut_range / resolution)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask.astype(bool)
```

**Step 3：在 KD-tree 构建时使用掩码**

```python
class AdaptiveKDTree:
    def __init__(self, ..., trajectory_mask=None):
        self.trajectory_mask = trajectory_mask
        # ...
    
    def _should_subdivide_by_features(self, node: KDTreeNode, feature_stats: dict) -> bool:
        """
        细分判断：加入轨迹掩码检查
        """
        # 先检查节点是否在轨迹区域内
        if self.trajectory_mask is not None:
            if not self._node_in_trajectory_region(node):
                return False  # 不在轨迹区域，不细分
        
        # 计算特征强度
        intensity = self._calculate_feature_intensity(node)
        
        return intensity > self.feature_threshold
    
    def _node_in_trajectory_region(self, node: KDTreeNode) -> bool:
        """
        检查节点是否在轨迹区域内
        """
        # 将节点坐标转换为掩码坐标
        # 检查节点中心或边界是否与掩码重叠
        center = node.center
        # ... 坐标转换逻辑
        
        # 简化版：检查节点中心是否在掩码内
        mask_x = int((center[0] - self.min_coords[0]) / self.resolution)
        mask_y = int((center[1] - self.min_coords[1]) / self.resolution)
        
        if 0 <= mask_x < self.trajectory_mask.shape[0] and \
           0 <= mask_y < self.trajectory_mask.shape[1]:
            return self.trajectory_mask[mask_x, mask_y]
        
        return False
```

**Step 4：使用示例**

```python
# 在 demo_multiview_bev_kdtree.py 中

# 1. 获取车辆轨迹
poses_xy = dataset.camera2world_all[:, :2, 3]  # (N, 2)

# 2. 创建轨迹掩码
trajectory_mask = create_trajectory_mask(
    poses_xy, 
    min_xy, 
    max_xy,
    resolution=0.05,
    cut_range=5.0  # 轨迹周围 5 米
)

print(f"轨迹掩码覆盖率: {trajectory_mask.sum() / trajectory_mask.size * 100:.1f}%")

# 3. 构建 KD-tree（带轨迹掩码）
tree = AdaptiveKDTree(
    min_area=0.0025,
    max_depth=18,
    feature_threshold=0.2,
    trajectory_mask=trajectory_mask,  # 传入掩码
    trajectory_mask_resolution=0.05,
    trajectory_mask_min_coords=min_xy
)

tree.build_from_boundary_points(xyz_all[:, :2], feature_map=sem_feat)
```

---

## 三、性能优化效果预估

### 当前性能
- **场景尺寸**: 208m × 40m = 8,410 m²
- **点数**: 2,220,782
- **图像**: 120 张（6 相机 × 20 帧）
- **语义融合时间**: ~15 秒
- **KD-tree 构建时间**: ~6 秒
- **总时间**: ~21 秒

### 优化后（方案 A + B）

#### 方案 A：简化细分策略
- 移除多条件判断，只计算一次特征强度
- 预期加速：**2-3倍**
- KD-tree 构建时间：~2-3 秒

#### 方案 B：轨迹预处理
- 假设轨迹掩码覆盖 30% 区域（车辆行驶范围）
- 细分区域减少：70%
- KD-tree 构建时间：~1-2 秒

#### 综合效果
- **总时间**: ~16-18 秒（vs 21 秒）
- **加速比**: ~1.2-1.3倍
- **顶点数**: 更合理的分布（复杂区域细，简单区域粗）

---

## 四、推荐实施步骤

### 第 1 步：实现简化细分策略（高优先级）

1. 修改 `models/kdtree.py`
2. 实现 `_calculate_feature_intensity`（选择选项 1 或 2）
3. 简化 `_should_subdivide_by_features`
4. 测试并验证顶点数分布

### 第 2 步：添加轨迹预处理（中优先级）

1. 实现 `create_trajectory_mask` 函数
2. 修改 `AdaptiveKDTree.__init__` 接受掩码参数
3. 在 `_should_subdivide_by_features` 中添加掩码检查
4. 测试加速效果

### 第 3 步：参数调优（低优先级）

1. 调整 `feature_threshold`（0.1 - 0.3）
2. 调整 `cut_range`（3-10 米）
3. 可视化不同参数下的网格划分结果

---

## 五、代码模板

我会为你创建：
1. ✅ `models/kdtree_v2.py` - 简化版 KD-tree
2. ✅ `utils/trajectory_utils.py` - 轨迹预处理工具
3. ✅ `demo_kdtree_v2.py` - 使用新策略的完整 demo

是否开始实现？
