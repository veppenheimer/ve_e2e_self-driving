import onnx

onnx_path = "results/ve.onnx"
onnx_model = onnx.load(onnx_path)

# 检查 ONNX 结构是否正确
try:
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 模型验证通过！")
except onnx.checker.ValidationError as e:
    print("❌ ONNX 模型验证失败：", e)
