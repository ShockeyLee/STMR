# qwen_vl_api_detector.py
# -*- coding: utf-8 -*-

import torch
import re
import requests
import base64
import mimetypes
import cv2
from typing import List, Tuple,Union 
import os

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-xxxx") # 替换为您的API密钥
API_URL = "https://api.siliconflow.cn/v1/chat/completions"


class QwenVLAPIDetector:
    """
    Encapsulates the Qwen-VL model via the SiliconFlow API for open-vocabulary object detection.
    (ROBUST VERSION: Handles inconsistent formats and multiple detections per prompt)
    """
    def __init__(self, api_key: str = SILICONFLOW_API_KEY, model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct", api_url: str = API_URL):
        if not api_key:
            raise ValueError("API key is required for the detector.")
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print(f"Qwen-VL API Detector initialized for model: {self.model_name}")

    # <--- MODIFICATION 1: Function renamed and logic completely rewritten for robustness
    def _parse_bounding_boxes(self, response_text: str) -> List[Tuple[int, int, int, int]]:
        """
        Parses all bounding boxes from the model's text response, handling various formats.
        It finds all <box>...</box> tags and extracts numbers from within.
        """
        found_boxes = []
        # Find all content within <box>...</box> tags
        box_contents = re.findall(r'<box>(.*?)</box>', response_text)
        
        for content in box_contents:
            # Find all numerical digits within the content
            numbers = re.findall(r'\d+', content)
            # If we find exactly 4 numbers, we assume it's a valid box
            if len(numbers) == 4:
                try:
                    x1, y1, x2, y2 = map(int, numbers)
                    found_boxes.append((x1, y1, x2, y2))
                except ValueError:
                    # This handles cases where numbers might be malformed, though unlikely with \d+
                    continue
        return found_boxes

    def predict(self, image_path: str, text_prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Performs object detection for each text prompt by querying the API.
        """
        all_boxes, all_scores, all_phrases = [], [], []

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(image_path)
            data_uri = f"data:{mime_type or 'image/jpeg'};base64,{base64_image}"
        except Exception as e:
            print(f"Error reading or encoding image file: {e}")
            return torch.empty(0, 4), torch.empty(0), []
        
        print(f"Contacting Qwen-VL API to detect {len(text_prompts)} landmarks...")
        for prompt in text_prompts:
            # We now ask for potentially multiple objects to encourage list-style responses
            detection_prompt = f"Find all instances of '{prompt}' in the image. For each instance, respond with its bounding box in the format <box>(x1,y1),(x2,y2)>."
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}, {"type": "text", "text": detection_prompt}]}],
                "temperature": 0.1,
                "max_tokens": 256 # Increased max_tokens to allow for multiple boxes
            }

            try:
                response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                
                # <--- MODIFICATION 2: Call the new parsing function
                parsed_boxes = self._parse_bounding_boxes(content)
                
                # <--- MODIFICATION 3: Handle the list of returned boxes
                if parsed_boxes:
                    print(f"  - SUCCESS: Detected {len(parsed_boxes)} instance(s) for '{prompt}'")
                    for box in parsed_boxes:
                        all_boxes.append(list(box))
                        all_scores.append(1.0)
                        all_phrases.append(prompt) # Each box gets the same phrase label
                else:
                    print(f"  - FAILED: Could not parse any valid box for '{prompt}'. Raw response: {content[:100]}...")

            except requests.RequestException as e:
                print(f"  - ERROR: API request failed for prompt '{prompt}': {e}")
            except (KeyError, IndexError) as e:
                print(f"  - ERROR: Failed to parse API response for prompt '{prompt}': {e} - Response: {response.text}")
        
        if not all_boxes:
            print("\nNo objects were detected across all prompts.")
            return torch.empty(0, 4), torch.empty(0), []

        print(f"\nTotal objects detected: {len(all_boxes)}")
        return torch.tensor(all_boxes, dtype=torch.float32), torch.tensor(all_scores, dtype=torch.float32), all_phrases