# grounding_dino_model.py
# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from GroundingDINO.groundingdino.util import box_ops
from typing import List, Tuple

class GroundingDINOModel:
    """
    封装Grounding DINO模型，使其易于加载和进行推理。
    """
    def __init__(self, config_path: str, weights_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载 Grounding DINO 模型到 {self.device}...")
        self.model = load_model(config_path, weights_path)
        self.model.to(self.device)
        print("Grounding DINO 模型加载完毕。")

    def predict(self, image_path: str, text_prompts: List[str], box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        对单个图像和文本提示列表进行目标检测。

        参数:
            image_path (str): 图像文件路径。
            text_prompts (List[str]): 用于检测的文本提示列表。
            box_threshold (float): 边界框置信度阈值。
            text_threshold (float): 文本相关性阈值。

        返回:
            Tuple[torch.Tensor, torch.Tensor, List[str]]:
            - boxes: 检测到的边界框 (归一化的xyxy格式)。
            - scores: 对应的置信度分数。
            - phrases: 每个框对应的文本短语。
        """
        try:
            # Grounding DINO内部的load_image已经包含了ToTensor和归一化
            image_source, image_tensor = load_image(image_path)
            
            # 将所有文本提示合并为一个字符串，用'.'分隔，这是DINO的标准做法
            text_prompt_str = " . ".join(text_prompts)
            
            print(f"Grounding DINO 正在检测: '{text_prompt_str}'")

            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt_str,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )

            # 获取图像尺寸用于坐标转换
            h, w, _ = image_source.shape
            # 将中心点+宽高格式(cxcywh)转换为角点格式(xyxy)
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
            # 归一化坐标
            # boxes_normalized = boxes_xyxy / torch.Tensor([w, h, w, h])

            print(f"检测到 {len(boxes)} 个物体。")
            return boxes_xyxy, logits, phrases

        except Exception as e:
            print(f"Grounding DINO 推理时发生错误: {e}")
            return torch.empty(0), torch.empty(0), []