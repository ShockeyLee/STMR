# run_pipeline.py (更新版)
# -*- coding: utf-8 -*-

# 导入新的模块
import cv2
import numpy as np
from llm_extractor import extract_landmarks_qwen
from grounding_dino_model import GroundingDINOModel
from tap_model import TokenizeAnythingModel
from visualize_matrix import visualize_stmr_matrix
from visualizer import visualize_output
from point_cloud_processor import read_pfm, project_masks_to_world,Pose
from detect_with_qwen import QwenVLAPIDetector
from stmr_model import STMR_Map

def main():
    """
    主函数，负责初始化模型并运行完整的感知流程，包括3D投影。
    """
    # --- 用户配置区 ---
    # Grounding DINO 模型配置
    # DINO_CONFIG_PATH = "configs/GroundingDINO_SwinT_OGC.py"
    # DINO_WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    
    # Tokenize Anything 权重路径
    TAP_WEIGHTS_PATH = "tokenize-anything/weights/tap_vit_l_v1_0.pkl"
    CONCEPT_WEIGHTS_PATH = "tokenize-anything/weights/merged_2560.pkl" 
    # AirSim 相机内参 (示例值，请替换为您的真实值)
    # 假设图像分辨率为 640x480, FoV为90度
    img_width, img_height = 1280, 960
    fov_degrees = 90
    focal_length = img_width / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    CAMERA_INTRINSICS = {
        'fx': focal_length, 'fy': focal_length,
        'cx': img_width / 2,  'cy': img_height / 2
    }
    DRONE_POSE_PATH = "drone_pose.json"
    # 测试数据
    TEST_IMAGE_PATH = "assert/test.png"     # ！！！请替换为您的RGB图像路径
    TEST_DEPTH_PATH = "assert/test.pfm"  # ！！！请替换为您的PFM深度图路径
    TEST_INSTRUCTION = "Take off in front of the light-brown two-story building on the right, then fly straight along the main street, passing the parking lot with cars on your left. Continue forward, keeping altitude as you approach the rounded brown high-rise. Slightly adjust left, and move toward the cluster of gray and black facade buildings at the end of the road, then stop above the street between them."
    # --- 配置结束 ---

    # 1. 初始化所有模型
    print("="*50)
    print("正在初始化所有模型...")
    # dino_model = GroundingDINOModel(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
    qwen_detector = QwenVLAPIDetector()
    tap_model = TokenizeAnythingModel(checkpoint=TAP_WEIGHTS_PATH,concept_weights=CONCEPT_WEIGHTS_PATH)
    print("所有模型初始化完毕！\n")

    # 2. 运行流程
    print("="*50)
    print("--- 开始运行完整感知流程 ---")
    
    # 步骤 2.1: 提取地标
    landmarks = extract_landmarks_qwen(TEST_INSTRUCTION)
    if not landmarks:
        print("流程终止: 未从指令中提取到任何地标。")
        return

    # 步骤 2.2: 目标检测
    boxes, scores, phrases = qwen_detector.predict(TEST_IMAGE_PATH, landmarks)
    if boxes.numel() == 0:
        print("流程终止: Grounding DINO 未检测到任何物体。")
        return
        
    # 步骤 2.3: 生成2D掩码
    image = cv2.imread(TEST_IMAGE_PATH)
    masks = tap_model.predict(image, boxes)
    if not masks:
        print("流程终止: Tokenize Anything 未能生成掩码。")
        return
        
    # 步骤 2.4 (新增): 读取深度图并进行3D投影
    print("\n--- 开始步骤 2.4: 3D投影 ---")
    try:
        depth_map = read_pfm(TEST_DEPTH_PATH)
        print(f"成功读取深度图，尺寸: {depth_map.shape}")
        drone_pose = Pose.from_json(DRONE_POSE_PATH)
        # 调用投影函数
        semantic_point_cloud = project_masks_to_world(masks, phrases, depth_map, CAMERA_INTRINSICS,drone_pose,0,250)
        # 打印一些点云信息作为验证
        if semantic_point_cloud:
            print("\n3D语义点云生成完毕。示例:")
            for item in semantic_point_cloud:
                label = item['label']
                points = item['points']
                
                # 打印每个地标的前3个点作为抽样检查
                print(f"  地标 '{label}': {points.shape[0]} 个点。前3个点坐标(X,Y,Z):\n{points[:3]}")
        else:
            print("未能生成任何3D点云数据。")

    except FileNotFoundError:
        print(f"错误: 无法找到深度图文件 '{TEST_DEPTH_PATH}'")
        print("请确保路径正确。")
        return
    except Exception as e:
        print(f"处理深度图或进行3D投影时发生错误: {e}")
        return

    # 步骤 2.5: 可视化2D结果
    visualize_output(TEST_IMAGE_PATH, boxes, masks, phrases)
    
    # 初始化STMR地图
    stmr_map = STMR_Map(map_size_meters=500, resolution=0.5)
    print("所有模型初始化完毕！\n")
    
    # ===================================================================
    # 新增：步骤 4 - 构建语义表征并生成最终的语义矩阵
    # ===================================================================
    print("\n--- 开始步骤 4: 构建语义表征并生成语义矩阵 ---")

    # C1: 使用世界坐标点云更新全局地图
    stmr_map.update_map(semantic_point_cloud)

    # C2: 根据当前子目标，生成最新的2D顶视图
    current_sub_goal_labels = [label] # 示例
    stmr_map.generate_top_down_view(current_sub_goal_labels=phrases)

    # C3: 获取无人机当前在世界地图上的位置 (X_north, Y_east)
    uav_current_position_xy = (
        drone_pose.position[0], # World X (North)
        drone_pose.position[1]  # World Y (East)
    )
    
    # C4: 提取LLM所需的局部语义矩阵
    semantic_matrix = stmr_map.get_local_matrix_representation(
        uav_world_pos_xy=uav_current_position_xy,
        local_map_size_meters=150,
        matrix_size=25
    )
    
    print("\n" + "="*25)
    print("  最终生成的 20x20 语义矩阵  ")
    print("="*25)
    print(semantic_matrix)
    
    # 为了让LLM理解这个矩阵，我们还需要提供ID到标签的映射
    print("\n语义ID到地标名称的映射:")
    # 反转字典以便于阅读
    id_to_label = {v: k for k, v in stmr_map.label_to_id.items()}
    print(id_to_label)
    
    print("\n--- 正在生成可视化图像... ---")
    visualize_stmr_matrix(semantic_matrix, id_to_label)
    
    print("\n--- 流程成功执行完毕 ---")
    print("="*50)

if __name__ == "__main__":
    main()