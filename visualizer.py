# visualizer.py
# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
from typing import List

def visualize_output(image_path: str, boxes: torch.Tensor, masks: List[np.ndarray], labels: List[str], output_filename: str = "output_visualization.jpg"):
    """在图像上绘制边界框和掩码以便于观察。"""
    image = cv2.imread(image_path)
    vis_image = image.copy()
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]

    for i, (box, mask, label) in enumerate(zip(boxes, masks, labels)):
        color = colors[i % len(colors)]
        
        # 绘制半透明的掩码
        mask_colored = np.zeros_like(vis_image, dtype=np.uint8)
        mask_colored[mask == 1] = color
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, 0.5, 0)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = colors[i % len(colors)]
        
        # 绘制边界框
        h, w, _ = image.shape
        box_unnorm = [int(b) for b in (box.cpu() * torch.tensor([w, h, w, h]))]
        x1, y1, x2, y2 = box_unnorm
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1 - 10), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_filename, vis_image)
    print(f"可视化结果已保存至 '{output_filename}'")