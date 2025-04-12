import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# TensorRT 记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加载 TensorRT 引擎
engine_file = "results/autodrive_fp16.engine"

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(engine_file)

# 创建 TensorRT 运行上下文
context = engine.create_execution_context()

# 绑定输入/输出
input_binding_idx = engine.get_binding_index("input")
output_binding_idx = engine.get_binding_index("output")

# 分配 GPU 内存
input_shape = (1, 3, 120, 160)  # (batch, C, H, W)
input_mem = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
output_mem = cuda.mem_alloc(4)  # 假设输出是 1 个 steering angle

# CUDA 流
stream = cuda.Stream()

# 预处理图像
img_path = "results/test.jpg"  # 你的测试图像路径
img = cv2.imread(img_path)
img = cv2.resize(img, (160, 120))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
img = np.expand_dims(img, axis=0)

# 复制数据到 GPU
cuda.memcpy_htod_async(input_mem, img, stream)

# 开始计时
start_time = time.time()

# 推理（异步执行）
context.execute_async_v2([int(input_mem), int(output_mem)], stream.handle, None)

# 复制结果回 CPU
output = np.empty(1, dtype=np.float32)
cuda.memcpy_dtoh_async(output, output_mem, stream)

# 同步等待所有操作完成
stream.synchronize()

# 结束计时
end_time = time.time()
inference_time = (end_time - start_time) * 1000.0  # 毫秒

print(f"TensorRT 推理结果: {output[0]}")
print(f"单次推理耗时: {inference_time:.2f} 毫秒")
