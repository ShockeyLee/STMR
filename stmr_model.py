# stmr_model.py (修正版，适配NED世界坐标系)
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

class STMR_Map:
    """
    管理语义-拓扑-度量表征 (STMR)。
    该类现在直接在NED世界坐标系下构建地图。
    """
    def __init__(self, map_size_meters: int = 500, resolution: float = 0.5):
        self.map_size_meters = map_size_meters
        self.resolution = resolution
        self.map_size_pixels = int(map_size_meters / resolution)
        self.map_origin_offset = self.map_size_meters / 2.0
        self.global_voxel_map: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
        self.global_top_down_map = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8)
        self.label_to_id: Dict[str, int] = {"unexplored": 0}
        self.next_label_id = 1
        print(f"STMR地图已初始化 (NED坐标系): {self.map_size_pixels}x{self.map_size_pixels} 像素, 分辨率 {self.resolution} m/px.")

    def _get_or_create_label_id(self, label: str) -> int:
        if label not in self.label_to_id:
            self.label_to_id[label] = self.next_label_id
            self.next_label_id += 1
        return self.label_to_id[label]

    def _world_to_map_coords(self, points_3d: np.ndarray) -> np.ndarray:
        """
        将世界坐标 (NED: X-North, Y-East) 批量转换为地图像素坐标 (u-col, v-row)。
        """
        # Y_east (世界坐标Y) -> u (地图列坐标，向右为正)
        u = ((points_3d[:, 1] + self.map_origin_offset) / self.resolution).astype(int)
        # X_north (世界坐标X) -> v (地图行坐标，向上为正，但数组索引向下为正，故取反)
        v = ((-points_3d[:, 0] + self.map_origin_offset) / self.resolution).astype(int)
        return np.vstack((u, v)).T

    def update_map(self, semantic_point_cloud_world: List[Dict[str, Any]]):
        """使用新的、世界坐标系(NED)下的语义点云来更新全局体素地图。"""
        print("--- 正在使用新的点云更新全局STMR体素地图 (NED) ---")
        for item in semantic_point_cloud_world:
            label, points_3d = item['label'], item['points']
            if points_3d.shape[0] == 0: continue
            semantic_id = self._get_or_create_label_id(label)
            map_coords = self._world_to_map_coords(points_3d)
            for i, (u, v) in enumerate(map_coords):
                if 0 <= u < self.map_size_pixels and 0 <= v < self.map_size_pixels:
                    z_down = points_3d[i, 2] # Z坐标在NED中是向下的
                    map_key = (v, u)
                    if map_key not in self.global_voxel_map:
                        self.global_voxel_map[map_key] = []
                    # 注意：因为Z向下，所以Z值越小代表高度越高。投影时应选择Z值最小的。
                    self.global_voxel_map[map_key].append((z_down, semantic_id))
        print(f"全局体素地图更新完毕。")

    def generate_top_down_view(self, current_sub_goal_labels: List[str] = []):
        """根据体素数据和当前子目标，生成最新的2D自顶向下地图。"""
        print("--- 正在生成新的2D自顶向下视图 (NED) ---")
        sub_goal_ids = {self._get_or_create_label_id(label) for label in current_sub_goal_labels}
        for (v, u), voxels in self.global_voxel_map.items():
            if not voxels: continue
            sub_goal_voxels = [voxel for voxel in voxels if voxel[1] in sub_goal_ids]
            # 因为Z向下，所以Z值最小的代表最高处的物体
            if sub_goal_voxels:
                best_voxel = min(sub_goal_voxels, key=lambda item: item[0])
            else:
                best_voxel = min(voxels, key=lambda item: item[0])
            self.global_top_down_map[v, u] = best_voxel[1]
        print("2D自顶向下视图生成完毕。")

    def get_local_matrix_representation(
        self,
        uav_world_pos_xy: Tuple[float, float], # (X_north, Y_east)
        local_map_size_meters: int = 100,
        matrix_size: int = 20
    ) -> np.ndarray:
        """从全局地图中提取一个以无人机为中心的局部矩阵。"""
        x_north, y_east = uav_world_pos_xy
        # 将无人机世界坐标转换为全局地图像素坐标
        uav_map_u = int((y_east + self.map_origin_offset) / self.resolution)
        uav_map_v = int((-x_north + self.map_origin_offset) / self.resolution)

        local_map_size_pixels = int(local_map_size_meters / self.resolution)
        half_size = local_map_size_pixels // 2
        
        v_start, v_end = uav_map_v - half_size, uav_map_v + half_size
        u_start, u_end = uav_map_u - half_size, uav_map_u + half_size
        
        v_start, v_end = max(0, v_start), min(self.map_size_pixels, v_end)
        u_start, u_end = max(0, u_start), min(self.map_size_pixels, u_end)
        
        local_map_window = self.global_top_down_map[v_start:v_end, u_start:u_end]

        local_matrix = cv2.resize(
            local_map_window, (matrix_size, matrix_size), interpolation=cv2.INTER_NEAREST
        )
        print(f"已为无人机位置 (North:{x_north:.2f}, East:{y_east:.2f}) 生成 {matrix_size}x{matrix_size} 局部矩阵。")
        return local_matrix