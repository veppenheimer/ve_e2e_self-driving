import onnxruntime as ort
import numpy as np
import cv2
import torch

onnx_path = "results/vic_simplified.onnx"
img_path = "results/265_-0.6632.jpg"

# 读取并预处理图像
img = cv2.imread(img_path)  # 读取图片
img = cv2.resize(img, (160, 120))  # 调整大小（确保符合模型输入）
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 颜色空间转换（如果模型需要）

# 归一化处理
img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
img = np.transpose(img, (2, 0, 1))  # 调整通道顺序 (H, W, C) -> (C, H, W)
img = np.expand_dims(img, axis=0)  # 增加 batch 维度 (1, C, H, W)

# 加载 ONNX 运行时
ort_session = ort.InferenceSession(onnx_path)

# 进行推理
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_output = ort_session.run(None, ort_inputs)

print(f"ONNX 预测结果: {ort_output[0][0]}")
