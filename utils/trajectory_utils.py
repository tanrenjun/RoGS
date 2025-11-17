"""
车辆轨迹预处理工具

用于从数据集提取车辆轨迹并创建掩码，以加速 KD-tree 构建
参考 ROGS 原始的 cut_point_by_pose 方法
"""

import numpy as np
import cv2


def get_vehicle_trajectory(dataset):
    """
    从数据集提取车辆轨迹点
    
    Args:
        dataset: NuscDataset 实例
        
    Returns:
        poses_xy: (N, 2) 车辆在 BEV 平面的 x, y 坐标
    """
    # 方法1：从 camera2world 提取（相机位置即车辆位置）
    if hasattr(dataset, 'camera2world_all'):
        poses = dataset.camera2world_all  # (N, 4, 4)
        poses_xy = poses[:, :2, 3]  # 提取 x, y 平移分量
        return poses_xy
    
    # 方法2：从 chassis2world 提取
    elif hasattr(dataset, 'chassis2world_all'):
        poses = dataset.chassis2world_all
        poses_xy = poses[:, :2, 3]
        return poses_xy
    
    else:
        raise ValueError("Dataset does not have trajectory information")


def create_trajectory_mask(poses_xy, min_coords, max_coords, resolution=0.05, cut_range=5.0):
    """
    创建车辆轨迹附近的掩码
    
    参考 models/road.py 中的 cut_point_by_pose 方法
    
    Args:
        poses_xy: (N, 2) 车辆轨迹点坐标
        min_coords: BEV 最小坐标 [x_min, y_min]
        max_coords: BEV 最大坐标 [x_max, y_max]
        resolution: BEV 分辨率（米/像素）
        cut_range: 轨迹周围的范围（米），只在这个范围内进行细分
        
    Returns:
        mask: (H, W) 布尔掩码，True 表示需要细分的区域
        mask_shape: (H, W) 掩码形状
    """
    # 计算 BEV 图像尺寸
    bev_size_x = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
    bev_size_y = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))
    
    print(f"  创建轨迹掩码:")
    print(f"    - BEV 尺寸: {bev_size_x} × {bev_size_y} 像素")
    print(f"    - 分辨率: {resolution}m/pixel")
    print(f"    - 轨迹范围: ±{cut_range}m")
    
    # 将轨迹点转换为像素坐标
    pixel_xy = np.zeros_like(poses_xy)
    pixel_xy[:, 0] = (poses_xy[:, 0] - min_coords[0]) / resolution
    pixel_xy[:, 1] = (poses_xy[:, 1] - min_coords[1]) / resolution
    
    # 去重
    pixel_xy = np.unique(pixel_xy.round(), axis=0)
    
    # 裁剪到有效范围
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, bev_size_x - 1)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, bev_size_y - 1)
    pixel_xy = pixel_xy.astype(np.int32)
    
    # 创建初始掩码
    mask = np.zeros((bev_size_x, bev_size_y), dtype=np.uint8)
    mask[pixel_xy[:, 0], pixel_xy[:, 1]] = 1
    
    print(f"    - 轨迹点数: {len(pixel_xy)}")
    
    # 膨胀掩码（扩展到轨迹周围 cut_range 米）
    kernel_size = int(cut_range / resolution)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    coverage = mask.sum() / mask.size * 100
    print(f"    - 覆盖率: {coverage:.1f}%")
    
    return mask.astype(bool), (bev_size_x, bev_size_y)


def point_in_trajectory_region(point_xy, trajectory_mask, min_coords, resolution):
    """
    检查点是否在轨迹区域内
    
    Args:
        point_xy: (2,) 点的 BEV 坐标 [x, y]
        trajectory_mask: (H, W) 布尔掩码
        min_coords: BEV 最小坐标 [x_min, y_min]
        resolution: BEV 分辨率
        
    Returns:
        bool: True 如果点在轨迹区域内
    """
    # 转换为像素坐标
    pixel_x = int((point_xy[0] - min_coords[0]) / resolution)
    pixel_y = int((point_xy[1] - min_coords[1]) / resolution)
    
    # 检查边界
    if 0 <= pixel_x < trajectory_mask.shape[0] and \
       0 <= pixel_y < trajectory_mask.shape[1]:
        return trajectory_mask[pixel_x, pixel_y]
    
    return False


def region_overlaps_trajectory(min_coords_region, max_coords_region, 
                               trajectory_mask, min_coords_bev, resolution):
    """
    检查矩形区域是否与轨迹重叠
    
    Args:
        min_coords_region: 区域最小坐标 [x_min, y_min]
        max_coords_region: 区域最大坐标 [x_max, y_max]
        trajectory_mask: (H, W) 布尔掩码
        min_coords_bev: BEV 最小坐标
        resolution: BEV 分辨率
        
    Returns:
        bool: True 如果区域与轨迹重叠
    """
    # 将区域边界转换为像素坐标
    pixel_min_x = int((min_coords_region[0] - min_coords_bev[0]) / resolution)
    pixel_min_y = int((min_coords_region[1] - min_coords_bev[1]) / resolution)
    pixel_max_x = int((max_coords_region[0] - min_coords_bev[0]) / resolution)
    pixel_max_y = int((max_coords_region[1] - min_coords_bev[1]) / resolution)
    
    # 裁剪到有效范围
    pixel_min_x = max(0, pixel_min_x)
    pixel_min_y = max(0, pixel_min_y)
    pixel_max_x = min(trajectory_mask.shape[0], pixel_max_x)
    pixel_max_y = min(trajectory_mask.shape[1], pixel_max_y)
    
    # 检查区域内是否有任何像素在轨迹掩码中
    if pixel_min_x >= pixel_max_x or pixel_min_y >= pixel_max_y:
        return False
    
    region_mask = trajectory_mask[pixel_min_x:pixel_max_x, pixel_min_y:pixel_max_y]
    
    return np.any(region_mask)


def visualize_trajectory_mask(trajectory_mask, save_path=None):
    """
    可视化轨迹掩码
    
    Args:
        trajectory_mask: (H, W) 布尔掩码
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.imshow(trajectory_mask.T, origin='lower', cmap='gray', aspect='auto')
    plt.colorbar(label='Trajectory Region')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title(f'Vehicle Trajectory Mask (Coverage: {trajectory_mask.sum()/trajectory_mask.size*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轨迹掩码可视化已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()
