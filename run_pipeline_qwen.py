# run_pipeline.py
# -*- coding: utf-8 -*-

from llm_extractor import extract_landmarks_qwen
from detect_with_qwen import QwenVLAPIDetector
from tap_model import TokenizeAnythingModel
from visualizer import visualize_output
import cv2

def main():
    """
    主函数，负责初始化模型并运行完整的感知流程。
    """
    # --- 用户配置区 ---
    # Grounding DINO 模型配置
    # 请确保从官方仓库下载了配置文件和权重
    DINO_CONFIG_PATH = "configs/GroundingDINO_SwinT_OGC.py"
    DINO_WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    
    # Tokenize Anything 权重路径
    TAP_WEIGHTS_PATH = "tokenize-anything/weights/tap_vit_l_v1_0.pkl"
    
    # 测试数据
    TEST_IMAGE_PATH = "assert/test.png" # ！！！请务必替换为您的图片路径
    # TEST_INSTRUCTION = "Take off from the center of the field, rise, fly low clockwise along the red track keeping the red school in view, cross the roof to zoom in on the green buses,turn right toward the white building with the red roof, and finally ascend for a wide panorama."
    TEST_INSTRUCTION = "Take off in front of the light-brown two-story building on the right, then fly straight along the main street, passing the parking lot with cars on your left. Continue forward, keeping altitude as you approach the rounded brown high-rise. Slightly adjust left, and move toward the cluster of gray and black facade buildings at the end of the road, then stop above the street between them."
    # TEST_INSTRUCTION = "Instruction: Take off in front of the white houses on the right, fly straight along the main street, pass the parking lot with cars on your left, continue toward the tall buildings in the distance, and stop in front of the brown rounded high-rise."
    # --- 配置结束 ---

    # 1. 初始化所有模型
    print("="*50)
    print("正在初始化所有模型...")
    # dino_model = GroundingDINOModel(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
    qwen_detector = QwenVLAPIDetector()
    tap_model = TokenizeAnythingModel(TAP_WEIGHTS_PATH)
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
        
    # 步骤 2.3: 生成掩码
    masks = tap_model.predict(cv2.imread(TEST_IMAGE_PATH), boxes)
    if not masks:
        print("流程终止: Tokenize Anything 未能生成掩码。")
        return
        
    # 步骤 2.4: 可视化
    visualize_output(TEST_IMAGE_PATH, boxes, masks, phrases)
    
    print("--- 流程成功执行完毕 ---")
    print("="*50)

if __name__ == "__main__":
    main()