# -*- coding: utf-8 -*-
"""
Tokenize Anything (TAP) wrapper aligned with the reference inference notebook.
修复了缩放因子 (scale_xy) 的维度问题，避免了 IndexError。
- 增强了对 im_rescale 返回值的兼容性，支持 tuple、list、ndarray。
- 修复了掩膜始终为空的问题：确保掩膜阈值正确，并返回概率最高的 mask。
"""

import os
from typing import List, Optional

import cv2
import numpy as np
import torch


class TokenizeAnythingModel:
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "tap_vit_l",
        concept_weights: Optional[str] = None,
        device: Optional[str] = None,
        input_long_side: int = 1024,
    ) -> None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"未找到权重文件: {checkpoint}")

        try:
            from tokenize_anything import model_registry
            from tokenize_anything.utils.image import im_rescale, im_vstack
        except ImportError as e:
            raise ImportError(
                "请先安装 Tokenize Anything: pip install git+https://github.com/baaivision/tokenize-anything.git"
            ) from e

        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_long_side = int(input_long_side)

        self.model = model_registry[model_type](checkpoint=checkpoint)
        self.model.to(self.device)
        self.model.eval()

        if concept_weights:
            if not os.path.exists(concept_weights):
                raise FileNotFoundError(f"未找到概念投影权重: {concept_weights}")
            self.model.concept_projector.reset_weights(concept_weights)

        self._im_rescale = im_rescale
        self._im_vstack = im_vstack

    @torch.inference_mode()
    def predict(self, image: np.ndarray, boxes: torch.Tensor, mask_threshold: float = 0.5) -> List[np.ndarray]:
        if image is None or image.size == 0:
            raise ValueError("输入图像为空或无效。")
        if boxes is None or boxes.numel() == 0:
            return []
        if boxes.dim() != 2 or boxes.size(-1) != 4:
            raise ValueError("boxes 形状需为 [N,4]，坐标顺序为 (x1, y1, x2, y2)。")

        boxes = boxes.detach().to(torch.float32).cpu()
        H, W = image.shape[:2]
        img_list, img_scales = self._im_rescale(image, scales=[self.input_long_side], max_size=self.input_long_side)
        input_size, original_size = img_list[0].shape, (H, W)

        img_batch = self._im_vstack(
            img_list,
            fill_value=self.model.pixel_mean_value,
            size=(self.input_long_side, self.input_long_side),
        )
        inputs = self.model.get_inputs({"img": img_batch})
        inputs.update(self.model.get_features(inputs))

        scale_arr = np.array(img_scales).reshape(-1).astype("float32")
        if scale_arr.size == 1:
            scale_x = scale_y = float(scale_arr[0])
        elif scale_arr.size >= 2:
            scale_y, scale_x = float(scale_arr[0]), float(scale_arr[1])
        else:
            raise RuntimeError(f"无法解析缩放因子 img_scales: {img_scales}")

        boxes_np = boxes.numpy().astype("float32")
        boxes_np[:, [0, 2]] *= scale_x
        boxes_np[:, [1, 3]] *= scale_y

        inputs["boxes"] = boxes_np[None, ...]
        outputs = self.model.get_outputs(inputs)

        iou_pred: torch.Tensor = outputs["iou_pred"]
        mask_pred: torch.Tensor = outputs["mask_pred"]

        row_idx = torch.arange(iou_pred.shape[0], device=iou_pred.device)
        best_col = iou_pred.argmax(1)
        selected_masks = mask_pred[row_idx, best_col]  # (N, H', W')

        # 概率掩膜（未阈值化） -> 上采样 -> 截取 -> 回原图尺寸
        masks = self.model.upscale_masks(selected_masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = self.model.upscale_masks(masks, original_size).cpu()

        # 使用阈值化避免空掩膜
        masks = (masks[:, 0] >= mask_threshold).to(torch.uint8).numpy()
        return [m for m in masks]


__all__ = ["TokenizeAnythingModel"]
