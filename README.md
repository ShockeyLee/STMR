# 语义拓扑度量表征复现

# 基于carla+airsim的数据集构建

## 步骤
1. 地标提取器的构建
采用qwen的语言模型，设计提示词即可,具体实现参考extract_landmarks_qwen.py,使用了简单的"Qwen/Qwen3-8B"
任务简单，8B的模型可以handel提取任务
2. 