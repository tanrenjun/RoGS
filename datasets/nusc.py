import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import cv2
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from plyfile import PlyData
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from datasets.base import BaseDataset


def get_nusc_filted_color_map():
    """
    获取nuScenes简化语义标签的颜色映射表
    
    将重映射后的7类语义标签映射到RGB颜色，便于可视化：
    0: 掩码区域 (黑色)
    1: 车道线相关 (蓝色)  
    2: 路沿 (红色)
    3: 道路和人孔盖 (灰色)
    4: 人行道 (天蓝色)
    5: 地形 (浅绿色)
    6: 背景 (黄绿色)
    
    Returns:
        np.ndarray: 形状为(256, 1, 3)的颜色映射表
    """
    colors = np.zeros((256, 1, 3), dtype='uint8')
    colors[0, :, :] = [0, 0, 0]  # 掩码区域
    colors[1, :, :] = [0, 0, 255]  # 车道线相关
    colors[2, :, :] = [255, 0, 0]  # 路沿
    colors[3, :, :] = [211, 211, 211]  # 道路和人孔盖
    colors[4, :, :] = [0, 191, 255]  # 人行道
    colors[5, :, :] = [152, 251, 152]  # 地形
    colors[6, :, :] = [157, 234, 50]  # 背景
    return colors


def get_nusc_origin_color_map():
    """
    获取nuScenes原始64类语义标签的颜色映射表
    
    为nuScenes数据集的64个原始语义类别提供RGB颜色映射，
    这是在nuScenes官网中使用的标准颜色配置。
    
    Returns:
        np.ndarray: 形状为(256, 1, 3)的原始颜色映射表
    """
    colors = np.zeros((256, 1, 3), dtype='uint8')
    colors[0, :, :] = [165, 42, 42]  # 鸟
    colors[1, :, :] = [0, 192, 0]  # 地面动物
    colors[2, :, :] = [196, 196, 196]  # 路沿
    colors[3, :, :] = [190, 153, 153]  # 栅栏
    colors[4, :, :] = [180, 165, 180]  # 防护栏
    colors[5, :, :] = [90, 120, 150]  # 障碍物
    colors[6, :, :] = [102, 102, 156]  # 墙
    colors[7, :, :] = [128, 64, 255]  # 自行车道
    colors[8, :, :] = [140, 140, 200]  # 斑马线-普通
    colors[9, :, :] = [170, 170, 170]  # 路缘切割
    colors[10, :, :] = [250, 170, 160]  # 停车区
    colors[11, :, :] = [96, 96, 96]  # 行人区域
    colors[12, :, :] = [230, 150, 140]  # 轨道
    colors[13, :, :] = [128, 64, 128]  # 道路
    colors[14, :, :] = [110, 110, 110]  # 服务车道
    colors[15, :, :] = [244, 35, 232]  # 人行道
    colors[16, :, :] = [150, 100, 100]  # 桥梁
    colors[17, :, :] = [70, 70, 70]  # 建筑
    colors[18, :, :] = [150, 120, 90]  # 隧道
    colors[19, :, :] = [220, 20, 60]  # 人
    colors[20, :, :] = [255, 0, 0]  # 骑行者
    colors[21, :, :] = [255, 0, 100]  # 摩托车手
    colors[22, :, :] = [255, 0, 200]  # 其他骑手
    colors[23, :, :] = [200, 128, 128]  # 车道标线-斑马线
    colors[24, :, :] = [255, 255, 255]  # 车道标线-普通
    colors[25, :, :] = [64, 170, 64]  # 山
    colors[26, :, :] = [230, 160, 50]  # 沙地
    colors[27, :, :] = [70, 130, 180]  # 天空
    colors[28, :, :] = [190, 255, 255]  # 雪
    colors[29, :, :] = [152, 251, 152]  # 地形
    colors[30, :, :] = [107, 142, 35]  # 植被
    colors[31, :, :] = [0, 170, 30]  # 水
    colors[32, :, :] = [255, 255, 128]  # 横幅
    colors[33, :, :] = [250, 0, 30]  # 长椅
    colors[34, :, :] = [100, 140, 180]  # 自行车架
    colors[35, :, :] = [220, 220, 220]  # 广告牌
    colors[36, :, :] = [220, 128, 128]  # 排水井
    colors[37, :, :] = [222, 40, 40]  # CCTV摄像头
    colors[38, :, :] = [100, 170, 30]  # 消防栓
    colors[39, :, :] = [40, 40, 40]  # 接线盒
    colors[40, :, :] = [33, 33, 33]  # 邮箱
    colors[41, :, :] = [100, 128, 160]  # 人孔盖
    colors[42, :, :] = [142, 0, 0]  # 电话亭
    colors[43, :, :] = [70, 100, 150]  # 坑洞
    colors[44, :, :] = [210, 170, 100]  # 路灯
    colors[45, :, :] = [153, 153, 153]  # 电杆
    colors[46, :, :] = [128, 128, 128]  # 交通标志框
    colors[47, :, :] = [0, 0, 80]  # 电线杆
    colors[48, :, :] = [250, 170, 30]  # 交通灯
    colors[49, :, :] = [192, 192, 192]  # 交通标志(背面)
    colors[50, :, :] = [220, 220, 0]  # 交通标志(正面)
    colors[51, :, :] = [140, 140, 20]  # 垃圾桶
    colors[52, :, :] = [119, 11, 32]  # 自行车
    colors[53, :, :] = [150, 0, 255]  # 船
    colors[54, :, :] = [0, 60, 100]  # 公交车
    colors[55, :, :] = [0, 0, 142]  # 汽车
    colors[56, :, :] = [0, 0, 90]  # 房车
    colors[57, :, :] = [0, 0, 230]  # 摩托车
    colors[58, :, :] = [0, 80, 100]  # 轨道车辆
    colors[59, :, :] = [128, 64, 64]  # 其他车辆
    colors[60, :, :] = [0, 0, 110]  # 拖车
    colors[61, :, :] = [0, 0, 70]  # 卡车
    colors[62, :, :] = [0, 0, 192]  # 慢速轮式车辆
    colors[63, :, :] = [32, 32, 32]  # 车载设备
    colors[64, :, :] = [120, 10, 10]  # 本车
    return colors


def get_nusc_label_remaps():
    """
    获取nuScenes语义标签重映射表
    
    将原始的64类nuScenes语义标签重映射为7个简化类别：
    1: 车道线相关 (7, 8, 14, 23, 24)
    2: 路沿相关 (2, 9)  
    3: 道路和人孔盖 (41, 13)
    4: 人行道 (15)
    5: 地形 (29)
    6: 背景 (其他所有类别)
    
    Returns:
        np.ndarray: 形状为(256, 1)的重映射表
    """
    colors = np.ones((256, 1), dtype="uint8")
    colors *= 6  # 背景类别
    colors[7, :] = 1  # 车道标线
    colors[8, :] = 1  # 斑马线
    colors[14, :] = 1  # 服务车道
    colors[23, :] = 1  # 车道标线-斑马线
    colors[24, :] = 1  # 车道标线-普通
    colors[2, :] = 2  # 路沿
    colors[9, :] = 2  # 路缘切割
    colors[41, :] = 3  # 人孔盖
    colors[13, :] = 3  # 道路
    colors[15, :] = 4  # 人行道
    colors[29, :] = 5  # 地形
    return colors


def label2mask(label):
    """
    生成道路掩码
    
    根据语义标签生成道路区域的二进制掩码，将非道路区域标记为掩码。
    包括静态非道路区域（建筑、植被等）和动态物体（车辆、行人等）。
    
    Args:
        label: 语义标签数组，形状为(H, W)
        
    Returns:
        tuple: (mask, label) 
               - mask: 道路掩码，形状为(H, W)，道路区域为1，非道路区域为0
               - label: 处理后的标签，道路区域标签为64
    """
    # 静态非道路区域掩码
    # 包括: 天空、建筑、植被、水体、栅栏、墙、护栏、障碍物等
    mask = np.ones_like(label)
    label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                     | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | (
                             (30 <= label) & (label <= 40)) | (label >= 42)

    # 动态物体掩码（从车辆类别开始，ID >= 52）
    # 对动态物体进行2次膨胀操作，确保完全覆盖
    label_movable = label >= 52
    kernel = np.ones((10, 10), dtype=np.uint8)
    label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

    # 合并静态和动态非道路区域
    label_off_road = label_off_road | label_movable
    mask[label_off_road] = 0  # 非道路区域掩码为0
    
    # 将掩码区域标记为64（未知类别）
    label[~(mask.astype(bool))] = 64
    mask = mask.astype(np.float32)
    return mask, label


def loda_depth(depth_file):
    """
    加载稀疏深度图
    
    从npz格式文件中加载使用scipy稀疏矩阵存储的深度图
    
    Args:
        depth_file: 深度图文件路径
        
    Returns:
        scipy.sparse.csr_matrix: 稀疏深度图矩阵
    """
    loaded_data = np.load(depth_file)
    depth_img = sp.csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    return depth_img


def worldpoint2camera(points: np.ndarray, WH, cam2world, cam_intrinsic, min_dist: float = 1.0):
    """
    将世界坐标系中的点投影到相机图像平面
    
    这是点云到图像投影的核心函数，执行完整的坐标变换流程：
    世界坐标系 -> 相机坐标系 -> 图像坐标系
    
    Args:
        points: 世界坐标系中的3D点，形状为(N, 3)
        WH: 图像尺寸，(width, height)
        cam2world: 相机到世界的变换矩阵，形状为(4, 4)
        cam_intrinsic: 相机内参矩阵，形状为(3, 3)
        min_dist: 最小距离阈值，默认1.0米
        
    Returns:
        tuple: (uv, depths, mask)
               - uv: 图像坐标，形状为(2, N)，(u, v)坐标
               - depths: 对应的深度值，形状为(N,)
               - mask: 有效点掩码，形状为(N,)，True表示在图像范围内
    """
    width, height = WH
    # 变换矩阵求逆：从相机变换到世界 -> 世界变换到相机
    world2cam = np.linalg.inv(cam2world)  # (4, 4)
    
    # 世界坐标到相机坐标的变换
    points_cam = world2cam[:3, :3] @ points.T + world2cam[:3, 3:4]  # (3, N)
    depths = points_cam[2, :]  # 获取Z坐标作为深度 (N,)
    
    # 相机坐标到图像坐标的投影
    points_uv1 = view_points(points_cam, np.array(cam_intrinsic), normalize=True)  # (3, N)
    
    # 过滤条件：去除在相机后方、距离过近或超出图像边界的点
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)  # 距离过近的点
    mask = np.logical_and(mask, points_uv1[0, :] > 1)  # 超出左边界
    mask = np.logical_and(mask, points_uv1[0, :] < width - 1)  # 超出右边界
    mask = np.logical_and(mask, points_uv1[1, :] > 1)  # 超出上边界
    mask = np.logical_and(mask, points_uv1[1, :] < height - 1)  # 超出下边界
    
    # 提取有效点的坐标和深度
    uv = points_uv1[:, mask][:2, :]  # (2, N_valid)
    uv = np.round(uv).astype(np.uint16)  # 取整并转换为整数坐标
    depths = depths[mask]  # 对应的有效深度值
    return uv, depths, mask


class NuscDataset(BaseDataset):
    """
    nuScenes道路重建数据集类
    
    继承自BaseDataset，专门处理nuScenes数据集。
    支持多相机、LiDAR、语义分割和可选的深度信息。
    """
    
    def __init__(self, configs, use_label=True, use_depth=False):
        """
        初始化nuScenes数据集
        
        执行完整的数据加载和预处理流程，包括：
        - 初始化nuScenes API
        - 加载多场景数据
        - 处理相机和LiDAR数据
        - 生成地面真实点云
        - 执行数据验证和清理
        
        Args:
            configs: 配置字典，包含数据集路径、相机配置等
            use_label: 是否使用语义标签
            use_depth: 是否生成和返回深度信息
        """
        # 初始化nuScenes API
        self.nusc = NuScenes(version="v1.0-{}".format(configs["version"]), dataroot=configs["base_dir"], verbose=True)
        self.version = configs["version"]
        super().__init__()  # 调用基类初始化
        
        # 保存配置参数
        self.resized_image_size = (configs["image_width"], configs["image_height"])  # 图像尺寸
        self.base_dir = configs["base_dir"]  # 数据集基础目录
        self.image_dir = configs["image_dir"]  # 图像目录
        self.camera_names = configs["camera_names"]  # 相机名称列表
        self.min_distance = configs["min_distance"]  # 最小距离阈值
        clip_list = configs["clip_list"]  # 场景列表
        self.chassis2world_unique = []  # 唯一的车辆位姿
        self.raw_wh = dict()  # 原始图像尺寸
        
        # 数据使用标记
        self.use_label = use_label
        self.use_depth = use_depth
        self.lidar_times_all = []  # LiDAR时间戳
        self.lidar_filenames_all = []  # LiDAR文件路径
        self.lidar2world_all = []  # LiDAR位姿

        road_pointcloud = dict()  # 存储各场景的道路点云
        for scene_name in tqdm(clip_list, desc="Loading data clips"):
            # 获取该场景的所有样本记录
            records = [samp for samp in self.nusc.sample if self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            records.sort(key=lambda x: (x['timestamp']))  # 按时间戳排序

            print(f"Loading image from scene {scene_name}")
            # 加载相机数据
            cam_info, chassis_info = self.load_cameras(records)

            # 保存相机数据到完整数据集
            self.raw_wh[scene_name] = cam_info["wh"]
            self.camera2world_all.extend(cam_info["poses"])
            self.camera_times_all.extend(cam_info["times"])
            self.cameras_K_all.extend(cam_info["intrinsics"])
            self.cameras_idx_all.extend(cam_info["idxs"])
            self.image_filenames_all.extend(cam_info["filenames"])

            # 保存车辆数据
            self.chassis2world_unique.extend(chassis_info["unique_poses"])
            self.chassis2world_all.extend(chassis_info["poses"])

            # 生成标签文件路径（从图像路径转换）
            label_filenames = [rel_camera_path.replace("/CAM", "/seg_CAM").replace(".jpg", ".png") for rel_camera_path in cam_info["filenames"]]
            self.label_filenames_all.extend(label_filenames)

            # 加载LiDAR数据
            lidar_info = self.load_lidars(records)
            self.lidar_times_all.extend(lidar_info["times"])
            self.lidar_filenames_all.extend(lidar_info["filenames"])
            self.lidar2world_all.extend(lidar_info["poses"])

            # 加载地面真实点云
            point_gt_path = os.path.join(configs["road_gt_dir"], f"{scene_name}.ply")
            xyz, rgb, label = self.load_gt_points(point_gt_path)
            road_pointcloud[scene_name] = {"xyz": xyz, "rgb": rgb, "label": label}

        # 执行数据检查和验证
        self.file_check()
        if len(self.image_filenames_all) == 0:
            raise FileNotFoundError("No data found in the dataset")

        # 转换为numpy数组
        self.chassis2world_unique = np.array(self.chassis2world_unique)
        self.chassis2world_all = np.array(self.chassis2world_all)  # [N, 4, 4]
        self.camera2world_all = np.array(self.camera2world_all)  # [N, 4, 4]
        self.camera_times_all = np.array(self.camera_times_all)  # [N, ]

        self.lidar2world_all = np.array(self.lidar2world_all)  # [N, 4, 4]
        self.lidar_times_all = np.array(self.lidar_times_all)  # [N, ]

        # 参考位姿归一化：将第一个位姿作为参考点
        self.ref_pose = self.chassis2world_unique[0]
        ref_pose_inv = np.linalg.inv(self.ref_pose)
        self.chassis2world_unique = ref_pose_inv @ self.chassis2world_unique
        self.camera2world_all = ref_pose_inv @ self.camera2world_all
        self.chassis2world_all = ref_pose_inv @ self.chassis2world_all
        self.lidar2world_all = ref_pose_inv @ self.lidar2world_all

        # 对地面真实点云也进行归一化
        for scene_name in road_pointcloud.keys():
            xyz = road_pointcloud[scene_name]["xyz"]
            new_xyz = ref_pose_inv[:3, :3] @ xyz.T + ref_pose_inv[:3, 3:4]
            road_pointcloud[scene_name]["xyz"] = new_xyz.T

        # 合并所有场景的道路点云
        self.road_pointcloud = {k: np.concatenate([road_pointcloud[s][k] for s in road_pointcloud.keys()], axis=0) for k in ("xyz", "rgb", "label")}

        # 计算NeRF++归一化参数
        nerf_normalization = self.getNerfppNorm()
        self.cameras_extent = nerf_normalization["radius"]

    def __len__(self):
        return len(self.image_filenames_all)

    def __getitem__(self, idx):
        """
        获取指定索引的训练样本
        
        这是PyTorch Dataset的核心方法，返回单个训练样本的完整数据。
        包括图像预处理、内参调整、语义标签处理和可选的深度图生成。
        
        Args:
            idx: 样本索引
            
        Returns:
            dict: 包含以下键的训练样本字典
                  - "image": 归一化后的RGB图像 (H, W, 3)
                  - "idx": 样本索引
                  - "cam_idx": 相机索引 
                  - "image_name": 图像文件名
                  - "R": 相机外参的旋转矩阵 (3, 3)
                  - "T": 相机外参的平移向量 (3,)
                  - "K": 调整后的相机内参 (3, 3)
                  - "W": 图像宽度
                  - "H": 图像高度
                  - "mask": 道路掩码 (H, W)（如果使用标签）
                  - "label": 重映射后的语义标签 (H, W)（如果使用标签）
                  - "depth": 深度图 (H, W)（如果使用深度）
        """
        # 获取基本相机信息
        cam_idx = self.cameras_idx_all[idx]
        cam2world = self.camera2world_all[idx]
        K = self.cameras_K_all[idx]
        camera_name = self.camera_names[cam_idx]
        
        # 加载图像
        image_path = os.path.join(self.base_dir, self.image_filenames_all[idx])
        image_name = os.path.basename(image_path).split(".")[0]
        input_image = cv2.imread(image_path)

        # 图像预处理：裁剪天空部分和缩放
        crop_cy = int(self.resized_image_size[1] * 0.5)  # 裁剪行数
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # BGR转RGB
        resized_image = resized_image[crop_cy:, :, :]  # 裁剪天空
        gt_image = (np.asarray(resized_image) / 255.0).astype(np.float32)  # 归一化到[0,1]
        gt_image = np.clip(gt_image, 0.0, 1.0)  # 限制范围
        width, height = gt_image.shape[1], gt_image.shape[0]

        # 调整相机内参：考虑缩放和裁剪
        new_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        new_K[0, :] *= width_scale  # x方向缩放
        new_K[1, :] *= height_scale  # y方向缩放
        new_K[1][2] -= crop_cy  # 考虑裁剪的y偏移
        
        # 提取外参
        R = cam2world[:3, :3]  # 旋转矩阵
        T = cam2world[:3, 3]   # 平移向量

        # 构建基础样本字典
        sample = {"image": gt_image, "idx": idx, "cam_idx": cam_idx, "image_name": image_name, "R": R, "T": T, "K": new_K, "W": width, "H": height}

        # 处理语义标签
        if self.use_label:
            label_path = os.path.join(self.image_dir, self.label_filenames_all[idx])
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
            mask, label = label2mask(resized_label)
            
            # 后相机特殊处理：底部83%区域作为掩码
            if camera_name == "CAM_BACK":
                h = mask.shape[0]
                mask[int(0.83 * h):, :] = 0
            
            # 语义标签重映射和裁剪
            label = self.remap_semantic(label).astype(int)
            mask = mask[crop_cy:, :]
            label = label[crop_cy:, :]
            sample["mask"] = mask
            sample["label"] = label

        # 处理深度图
        if self.use_depth:
            # 找到最接近的LiDAR数据
            cam_time = self.camera_times_all[idx]
            lidar_idx = np.argmin(np.abs(self.lidar_times_all - cam_time))
            lidar2world = self.lidar2world_all[lidar_idx]
            
            # 加载LiDAR点云
            lidar_path = os.path.join(self.base_dir, self.lidar_filenames_all[lidar_idx])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            
            # 坐标变换：LiDAR到世界
            points_world = lidar2world[:3, :3] @ points.T + lidar2world[:3, 3:4]  # (3, N)
            
            # 投影到图像平面
            uv, depths, mask = worldpoint2camera(points_world.T, (width, height), cam2world, new_K)
            
            # 按深度排序（近到远）
            sort_idx = np.argsort(depths)[::-1]
            uv = uv[:, sort_idx]
            depths = depths[sort_idx]
            
            # 构建深度图
            depth_image = np.zeros((height, width), dtype=np.float32)
            depth_image[uv[1], uv[0]] = depths
            sample["depth"] = depth_image

        return sample

    def load_gt_points(self, ply_path):
        plydata = PlyData.read(ply_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)  # [N, 3]
        rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])), axis=1)
        label = np.asarray(plydata.elements[0]["label"]).astype(np.uint8)
        label = label[..., None]  # [N, 1]
        return xyz, rgb, label

    def load_lidars(self, records):
        lidar_times = []
        lidar_files = []
        lidar2worlds = []

        for rec in tqdm(records):
            samp = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
            flag = True
            while flag or not samp["is_key_frame"]:
                flag = False

                lidar_times.append(samp["timestamp"])
                lidar_files.append(samp["filename"])

                cs_record = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
                lidar2ego[:3, 3] = cs_record['translation']

                poserecord = self.nusc.get('ego_pose', samp['ego_pose_token'])
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(poserecord['rotation']).rotation_matrix
                ego2global[:3, 3] = poserecord['translation']

                lidar2global = ego2global @ lidar2ego
                lidar2worlds.append(lidar2global)

                if samp["next"] != "":
                    samp = self.nusc.get('sample_data', samp["next"])
                else:
                    break

        return {"times": lidar_times, "filenames": lidar_files, "poses": lidar2worlds}

    def load_cameras(self, records):
        chassis2world_unique = []
        chassis2worlds = []
        camera2worlds = []
        cameras_K = []
        cameras_idxs = []
        cameras_times = []
        image_filenames = []
        wh = dict()

        # interpolate images from 2HZ to 12 HZ  (sample + sweep)
        for rec in tqdm(records):
            chassis_flag = True
            for camera_idx, cam in enumerate(self.camera_names):
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = self.nusc.get("sample_data", rec_token)
                wh.setdefault(cam, (samp["width"], samp["height"]))
                flag = True
                while flag or not samp["is_key_frame"]:
                    flag = False
                    rel_camera_path = samp["filename"]
                    cameras_times.append(samp["timestamp"])
                    image_filenames.append(rel_camera_path)

                    camera2chassis = self.compute_extrinsic2chassis(samp)
                    c2w = self.compute_chassis2world(samp)
                    chassis2worlds.append(c2w)
                    if chassis_flag:
                        chassis2world_unique.append(c2w)
                    camera2world = c2w @ camera2chassis
                    camera2worlds.append(camera2world.astype(np.float32))

                    calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                    intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                    cameras_K.append(intrinsic.astype(np.float32))

                    cameras_idxs.append(camera_idx)
                    # not key frames
                    if samp["next"] != "":
                        samp = self.nusc.get('sample_data', samp["next"])
                    else:
                        break
                chassis_flag = False
        cam_info = {"poses": camera2worlds, "intrinsics": cameras_K, "idxs": cameras_idxs, "filenames": image_filenames, "times": cameras_times, "wh": wh}
        chassis_info = {"poses": chassis2worlds, "unique_poses": chassis2world_unique}

        return cam_info, chassis_info

    def compute_chassis2world(self, samp):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = self.nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global

    def compute_extrinsic(self, samp_a, samp_b):
        """transform from sensor_a to sensor_b"""
        sensor_a2chassis = self.compute_extrinsic2chassis(samp_a)
        sensor_b2chassis = self.compute_extrinsic2chassis(samp_b)
        sensor_a2sensor_b = np.linalg.inv(sensor_b2chassis) @ sensor_a2chassis
        return sensor_a2sensor_b

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array(Quaternion(calibrated_sensor["rotation"]).rotation_matrix)
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis

    def file_check(self):
        image_paths = [os.path.join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        print(f"Drop {len(image_paths) - len(np.where(image_exists)[0])} frames out of {len(image_paths)} by image exists check")
        exists = image_exists
        label_paths = [os.path.join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        print(f"Drop {len(image_paths) - len(np.where(label_exists)[0])} frames out of {len(image_paths)} by label exists check")
        exists *= label_exists

        lidar_paths = [os.path.join(self.base_dir, lidar_path) for lidar_path in self.lidar_filenames_all]
        lidar_exists = np.asarray(self.check_filelist_exist(lidar_paths))
        print(f"Drop {len(lidar_paths) - len(np.where(lidar_exists)[0])} lidar out of {len(lidar_paths)} by lidar exists check")
        lidar_available = list(np.where(lidar_exists)[0])
        self.lidar_times_all = [self.lidar_times_all[i] for i in lidar_available]
        self.lidar_filenames_all = [self.lidar_filenames_all[i] for i in lidar_available]
        self.lidar2world_all = [self.lidar2world_all[i] for i in lidar_available]

        available_index = list(np.where(exists)[0])
        print(f"Drop {len(image_paths) - len(available_index)} frames out of {len(image_paths)} by file exists check")
        self.filter_by_index(available_index)

    def label_valid_check(self):
        label_paths = [os.path.join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_valid = np.asarray(self.check_label_valid(label_paths))
        available_index = list(np.where(label_valid)[0])
        print(f"Drop {len(label_paths) - len(available_index)} frames out of {len(label_paths)} by label valid check")
        self.filter_by_index(available_index)

    def label_valid(self, label_name):
        label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
        label_movable = label >= 52
        ratio_movable = label_movable.sum() / label_movable.size
        label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                         | ((15 <= label) & (label <= 22)) | ((25 <= label) & (label <= 40)) | (label >= 42)
        ratio_static = label_off_road.sum() / label_off_road.size
        if ratio_movable > 0.3 or ratio_static > 0.9:
            return False
        else:
            return True

    def check_label_valid(self, filelist):
        with Pool(32) as p:
            exist_list = p.map(self.label_valid, filelist)
        return exist_list

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.camera2world_all = [self.camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        self.camera_times_all = [self.camera_times_all[i] for i in index]
        self.chassis2world_all = [self.chassis2world_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]

    @property
    def label_remaps(self):
        return get_nusc_label_remaps()

    @property
    def origin_color_map(self):
        return get_nusc_origin_color_map()

    @property
    def num_class(self):
        return 7

    @property
    def filted_color_map(self):
        return get_nusc_filted_color_map()
