import onnx
from onnx import helper

model = onnx.load("ve_0512_100b_simplified.onnx")
graph = model.graph

for input_tensor in graph.input:
    shape = input_tensor.type.tensor_type.shape
    # 将第一个维度（batch_size）改成固定值1
    if shape.dim[0].dim_param:  # 是动态维度
        shape.dim[0].dim_param = ""  # 清空动态参数
        shape.dim[0].dim_value = 1  # 固定为1

onnx.save(model, "ve_0512_100b_simplified_fixed.onnx")
