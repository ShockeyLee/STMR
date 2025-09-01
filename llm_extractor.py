import os
import json
import re  # 导入正则表达式模块
import requests
from typing import List, Optional, Dict, Any

# --- 配置区域 ---
# 建议通过环境变量设置API密钥，避免硬编码
# 您可以在终端执行: export SILICONFLOW_API_KEY="your_api_key_here"
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-xxxx") # 替换为您的API密钥
API_URL = "https://api.siliconflow.cn/v1/chat/completions"


def _extract_json_from_string(text: str) -> Optional[Dict[str, Any]]:
    """
    从可能包含额外文本的字符串中稳健地提取第一个JSON对象。

    Args:
        text (str): 模型返回的原始字符串。

    Returns:
        Optional[Dict[str, Any]]: 解析后的JSON对象（字典），如果找不到或解析失败则返回None。
    """
    # 使用正则表达式查找被```json ... ```包裹的代码块或直接的JSON对象
    # re.DOTALL 使得 '.' 可以匹配包括换行符在内的任意字符
    json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
    
    if not json_match:
        print(f"解析警告：在响应中未找到JSON格式的文本。")
        return None

    # 优先使用第一个捕获组（被```json包裹的），否则使用第二个（裸露的JSON）
    json_str = json_match.group(1) or json_match.group(2)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"解析错误：提取的字符串不是有效的JSON。错误: {e}")
        print(f"  -> 尝试解析的字符串: '{json_str}'")
        return None


def extract_landmarks_qwen(
    instruction: str, 
    api_key: str = SILICONFLOW_API_KEY, 
    model_name: str = "Qwen/Qwen3-14B"
) -> List[str]:
    """
    使用Qwen语言模型从导航指令中提取地标。

    该函数实现了论文中提到的由LLM驱动的“地标提取器 (Landmark Extractor LE(·))”。
    它向LLM发送一个精心设计的Prompt，要求其识别并以JSON列表的形式返回指令中的物理地标。

    Args:
        instruction (str): 用户提供的自然语言导航指令。
        api_key (str): 用于API认证的密钥。
        model_name (str): 要使用的模型名称。

    Returns:
        List[str]: 从指令中提取出的地标名称列表。如果提取失败或未找到地标，则返回空列表。
    """
    if not api_key or api_key == "sk-xxxx":
        print("错误：API密钥未设置。请设置SILICONFLOW_API_KEY环境变量或直接在代码中提供。")
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 精心设计的Prompt，指导模型执行地标提取任务
    system_prompt = """You are an AI assistant for a UAV (Unmanned Aerial Vehicle) navigation system. 
Your task is to extract key physical landmarks from a user's navigation instruction.

Instructions:
1.  Identify only physical, identifiable objects, structures, or geographical features that can be used for navigation.
2.  Examples of valid landmarks: 'road', 'red brick building', 'river', 'bridge', 'trees', 'yellow tractor', 'intersection','building with red roof'.
3.  Do NOT extract abstract concepts, directions (e.g., 'left', 'right', 'straight'), or actions (e.g., 'take off', 'turn', 'stop', 'go').
4.  Your output MUST be a single JSON object with a key named "landmarks", which contains a list of the extracted landmark strings.
5.  If no landmarks are found, return an empty list.

Example:
Instruction: "Take off, fly over the main street, then turn left towards the red building and stop."
Output: 
```json
{"landmarks": ["main street", "red building"]}
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"}
    }

    content_str = ""
    try:
        print(f"--- 向模型 '{model_name}' 发送请求 ---")
        print(f"指令: {instruction}")
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        response_json = response.json()
        content_str = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not content_str:
            print("错误：API响应中未找到有效内容。")
            return []

        # 使用更稳健的解析函数
        data = _extract_json_from_string(content_str)
        
        if data is None:
            print(f"解析失败，收到的原始响应内容: {content_str}")
            return []

        landmarks = data.get("landmarks", [])

        if isinstance(landmarks, list):
            print(f"成功提取地标: {landmarks}")
            return landmarks
        else:
            print(f"警告：模型返回的'landmarks'不是一个列表: {landmarks}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"响应内容: {e.response.text}")
    except (KeyError, IndexError) as e:
        print(f"解析API响应结构失败: {e}")
        print(f"收到的原始响应内容: {response.text if 'response' in locals() else 'N/A'}")
    except Exception as e:
        print(f"发生未知错误: {e}")

    return [] # 发生任何错误时返回空列表

if __name__ == "__main__":
    # 论文图3中的示例指令
    sample_instruction_1 = "Instruction: Rise from the rooftop and turn right. Fly over the middle of the street until you reach the buildings on the left with the short concrete wall like structure around them. Turn left right beside the traffic light and fly along the lane. When you get in side of the river then angle straight up and fly toward the top of the tower."

    # 论文图6中的示例指令
    sample_instruction_2 = "Instruction: Go along the road and slightly turn right in front of the trees, fly across the bridge, turn right beside the car and go forward, stop next to the building."

    # 简单的测试指令
    sample_instruction_3 = "从河流旁边起飞，然后左转，飞向那个黄色的工厂。"

    # 无地标的指令
    sample_instruction_4 = "起飞后直行100米。"

    print("--- 测试用例 1 ---")
    extracted_landmarks_1 = extract_landmarks_qwen(sample_instruction_1)
    print("-" * 20)

    print("\n--- 测试用例 2 ---")
    extracted_landmarks_2 = extract_landmarks_qwen(sample_instruction_2)
    print("-" * 20)

    print("\n--- 测试用例 3 (中文) ---")
    extracted_landmarks_3 = extract_landmarks_qwen(sample_instruction_3)
    print("-" * 20)

    print("\n--- 测试用例 4 (无地标) ---")
    extracted_landmarks_4 = extract_landmarks_qwen(sample_instruction_4)
    print("-" * 20)