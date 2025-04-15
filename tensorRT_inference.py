import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine_file = "results/20250410_simplified_fp16.engine"
img_path = "results/486_-1.0000.jpg"  # 测试图像路径


# 加载 TensorRT 引擎
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


engine = load_engine(engine_file)
context = engine.create_execution_context()


input_idx = engine.get_binding_index("input")
output_idx = engine.get_binding_index("output")

input_shape = (1, 3, 120, 160)
output_shape = (1,)

# 分配GPU显存
input_size = int(np.prod(input_shape) * np.float32().itemsize)
output_size = int(np.prod(output_shape) * np.float32().itemsize)

input_mem = cuda.mem_alloc(input_size)
output_mem = cuda.mem_alloc(output_size)

stream = cuda.Stream()

# 图像预处理
img = cv2.imread(img_path)
img = cv2.resize(img, (160, 120))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)  # HWC -> CHW
img = np.expand_dims(img, axis=0)  # 加 batch 维度
img = img.copy()



# 数据拷贝到GPU
cuda.memcpy_htod_async(input_mem, img, stream)

start_time = time.time()

# 推理
context.execute_async_v2(bindings=[int(input_mem), int(output_mem)], stream_handle=stream.handle)

# 从GPU取结果
output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output, output_mem, stream)

# 等待同步
stream.synchronize()

end_time = time.time()

print(f"TensorRT 推理结果: {output[0]}")
print(f"单次推理耗时: {(end_time - start_time) * 1000:.2f} 毫秒")
