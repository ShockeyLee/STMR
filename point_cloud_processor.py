# point_cloud_processor.py (根据您的函数进行重构和优化)
# -*- coding: utf-8 -*-

import numpy as np
import json
import re
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as ScipyRotation

class Pose:
    """
    一个用于管理和转换无人机位姿的类。
    """
    def __init__(self, position: np.ndarray, quaternion_wxyz: np.ndarray):
        """
        使用位置和四元数初始化位姿。

        参数:
            position (np.ndarray): (x, y, z) 位置向量。
            quaternion_wxyz (np.ndarray): (w, x, y, z) 格式的四元数。
        """
        self.position = position
        # Scipy使用 (x, y, z, w) 格式，我们进行转换
        q_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])
        self.rotation = ScipyRotation.from_quat(q_xyzw)

    @classmethod
    def from_json(cls, file_path: str):
        """从指定的JSON文件中加载位姿数据。"""
        print(f"--- 正在从 {file_path} 加载无人机位姿 ---")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        position = np.array([data['POS_X'], data['POS_Y'], data['POS_Z']])
        quaternion_wxyz = np.array([data['Q_W'], data['Q_X'], data['Q_Y'], data['Q_Z']])
        
        print(f"位置 (X,Y,Z): {position}")
        print(f"四元数 (W,X,Y,Z): {quaternion_wxyz}")
        return cls(position, quaternion_wxyz)

    def camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """
        将相机坐标系下的点云转换到世界坐标系。
        该函数严格遵循您提供的坐标变换逻辑。

        参数:
            points_camera (np.ndarray): Nx3 的点云数组，位于标准相机坐标系
                                       (X向右, Y向下, Z向前)。
        
        返回:
            np.ndarray: 转换到世界坐标系后的Nx3点云数组。
        """
        # 步骤 1: 将标准相机坐标系 (X right, Y down, Z forward) 
        #         调整为机体坐标系 (X forward, Y right, Z down)。
        # camera_coords[0] -> Xc (right)
        # camera_coords[1] -> Yc (down)
        # camera_coords[2] -> Zc (forward)
        # 调整后: [Zc, Xc, Yc] -> [X_body, Y_body, Z_body]
        points_body = points_camera[:, [2, 0, 1]]

        # 步骤 2: 应用旋转和平移 (P_world = R * P_body + T)
        rotation_matrix = self.rotation.as_matrix()
        
        # 使用 einsum 进行高效的批量矩阵乘法: (N, 3) @ (3, 3) -> (N, 3)
        points_world = np.einsum('ij,kj->ik', points_body, rotation_matrix) + self.position
        
        return points_world


def read_pfm(file_path: str) -> np.ndarray:
    """读取 .pfm 文件并返回一个包含深度数据的numpy数组。"""
    # (此函数保持不变)
    with open(file_path, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF': color = True
        elif header == 'Pf': color = False
        else: raise Exception('非法的PFM文件头')
        dims_line = file.readline().decode('utf-8').rstrip()
        dims_match = re.match(r'^(\d+)\s+(\d+)$', dims_line)
        width, height = map(int, dims_match.groups())
        scale = float(file.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.flipud(data.reshape(shape))


def project_masks_to_world(
    masks: List[np.ndarray],
    labels: List[str],
    depth_map: np.ndarray,
    camera_intrinsics: Dict[str, float],
    drone_pose: Pose,
    min_depth: float = 0.1,
    max_depth: float = 100.0
) -> List[Dict[str, Any]]:
    """
    将2D语义掩码直接投影到世界坐标系，生成带标签的点云。
    (此函数现在集成了从像素到世界的完整变换)

    参数:
        masks: 语义掩码列表。
        labels: 地标标签列表。
        depth_map: 深度图。
        camera_intrinsics: 相机内参, 包含 'fx', 'fy', 'cx', 'cy'。
        drone_pose (Pose): 包含无人机位姿的Pose对象。
        min_depth, max_depth: 深度过滤阈值。

    返回:
        List[Dict[str, Any]]: 世界坐标系下的语义点云列表。
    """
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    semantic_point_cloud_world = []
    
    print("--- 正在将2D掩码直接投影到世界坐标系 ---")
    
    for i, mask in enumerate(masks):
        v_coords, u_coords = np.where(mask > 0)
        if v_coords.size == 0:
            continue

        depth_values = depth_map[v_coords, u_coords]
        
        # 深度值过滤
        valid_depth_mask = (depth_values > min_depth) & (depth_values < max_depth)
        u_coords_valid = u_coords[valid_depth_mask]
        v_coords_valid = v_coords[valid_depth_mask]
        depth_values_valid = depth_values[valid_depth_mask]
        
        if len(v_coords_valid) == 0:
            continue
        
        # 步骤 1: 计算标准相机坐标系下的点 (Xc, Yc, Zc)
        z_c = depth_values_valid
        x_c = (u_coords_valid - cx) * z_c / fx
        y_c = (v_coords_valid - cy) * z_c / fy
        points_camera = np.vstack((x_c, y_c, z_c)).T

        # 步骤 2: 将相机坐标系下的点云转换到世界坐标系
        points_world = drone_pose.camera_to_world(points_camera)
        
        semantic_point_cloud_world.append({
            "label": labels[i],
            "points": points_world
        })
        
        print(f"  -> 已为地标 '{labels[i]}' 生成 {len(points_world)} 个世界坐标点。")
        
    return semantic_point_cloud_world