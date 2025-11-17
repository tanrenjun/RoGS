import os

import cv2
import numpy as np
from multiprocessing.pool import ThreadPool as Pool

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    道路重建数据集基类
    
    这是所有道路重建数据集的基类，提供了通用的数据管理功能。
    支持nuScenes和KITTI等多种数据集的统一处理接口。
    """
    
    def __init__(self):
        """
        初始化数据集基类
        
        初始化所有必要的数据容器，用于存储：
        - 图像和标签文件路径
        - 相机内外参信息
        - 相机位姿和时间戳
        - 其他辅助数据（如深度图等）
        """
        self.base_dir = ""  # 数据集基础目录
        
        # 基础数据属性（当前使用的数据子集）
        self.image_filenames = []  # 图像文件相对路径列表（相对于base_dir）
        self.label_filenames = []  # 标签文件相对路径列表（相对于base_dir）
        self.ref_camera2world = []  # 参考相机外参变换矩阵列表 (4x4 ndarray)
        self.cameras_K = []  # 相机内参矩阵列表 (3x3 ndarray)
        self.cameras_d = []  # 相机畸变系数列表
        self.cameras_idx = []  # 相机索引列表（用于标识不同相机）
        self.camera_times_all = []  # 相机时间戳列表

        # 完整数据集属性（所有数据）
        self.camera2world_all = []  # 所有相机外参变换矩阵列表 (4x4 ndarray)
        self.chassis2world_all = []  # 所有车辆底盘位姿变换矩阵列表 (4x4 ndarray)
        self.image_filenames_all = []  # 所有图像文件相对路径列表
        self.label_filenames_all = []  # 所有标签文件相对路径列表
        self.lane_filenames_all = []  # 车道线文件相对路径列表（可选）
        self.ref_camera2world_all = []  # 所有参考相机外参列表
        self.cameras_K_all = []  # 所有相机内参列表
        self.cameras_d_all = []  # 所有相机畸变系数列表
        self.cameras_idx_all = []  # 所有相机索引列表

    def __len__(self):
        """
        返回数据集的大小
        
        Returns:
            int: 当前使用的数据子集大小
        """
        return len(self.image_filenames)

    def getNerfppNorm(self):
        """
        计算用于NeRF++归一化的参数
        
        这是NeRF++论文中使用的归一化方法，将所有相机位置
        归一化到一个以原点为中心的范围内，便于神经辐射场的学习。
        
        Returns:
            dict: 包含'translate'和'radius'的字典
                  - translate: 平移向量，用于将场景中心移动到原点
                  - radius: 归一化半径，用于缩放场景
        """
        def get_center_and_diag(cam_centers):
            """
            计算所有相机位置的中心点和对角线距离
            
            Args:
                cam_centers: 所有相机位置的列表
                
            Returns:
                tuple: (center, diagonal) 中心和最大距离
            """
            cam_centers = np.hstack(cam_centers)  # 合并所有相机位置
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)  # 计算平均位置
            center = avg_cam_center  # 场景中心
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)  # 计算到中心的距离
            diagonal = np.max(dist)  # 最大距离作为对角线长度
            return center.flatten(), diagonal  # 返回展平的中心和距离

        # 提取所有相机的位置（从外参矩阵的第4列前3行得到）
        cam_centers = [pose[:3, 3:4] for pose in self.camera2world_all]

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1  # 稍微增加半径以确保所有点都被包含
        translate = -center  # 负中心坐标用于平移
        return {"translate": translate, "radius": radius}

    def filter_by_index(self, index):
        """
        根据索引过滤数据集
        
        从完整数据集中筛选出指定索引对应的数据，常用于数据采样、
        训练/验证集分割等场景。
        
        Args:
            index: 要保留的数据索引列表
        """
        # 过滤基础数据属性
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.lane_filenames_all = [self.lane_filenames_all[i] for i in index]
        self.ref_camera2world_all = [self.ref_camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_d_all = [self.cameras_d_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        
        # 如果存在深度数据，也进行过滤
        if hasattr(self, "depth_filenames_all"):
            self.depth_filenames_all = [self.depth_filenames_all[i] for i in index]

    @staticmethod
    def file_valid(file_name):
        """
        检查文件是否有效
        
        验证文件是否存在且非空，这是数据验证的重要步骤。
        
        Args:
            file_name: 要检查的文件路径
            
        Returns:
            bool: 文件存在且非空时返回True，否则返回False
        """
        if os.path.exists(file_name) and (os.path.getsize(file_name) != 0):
            return True
        else:
            return False

    @staticmethod
    def check_filelist_exist(filelist):
        """
        批量检查文件列表中文件的有效性
        
        使用多进程并行检查大量文件的有效性，提高效率。
        
        Args:
            filelist: 要检查的文件路径列表
            
        Returns:
            list: 与输入列表对应长度的布尔值列表，表示每个文件是否有效
        """
        with Pool(32) as p:  # 使用32个线程的线程池
            exist_list = p.map(BaseDataset.file_valid, filelist)  # 并行检查文件有效性
        return exist_list

    def remap_semantic(self, semantic_label):
        """
        重映射语义标签
        
        将原始语义标签映射到简化的标签类别，这是多类分类任务
        中的常见预处理步骤。
        
        Args:
            semantic_label: 原始语义标签数组
            
        Returns:
            np.ndarray: 重映射后的语义标签数组
        """
        semantic_label = semantic_label.astype('uint8')  # 确保数据类型正确
        # 使用OpenCV的LUT查表功能进行快速重映射
        remaped_label = np.array(cv2.LUT(semantic_label, self.label_remaps))
        return remaped_label
