### nano板子中安装pycuda库

pip install pycuda

### 生成tensorRT引擎

sudo apt-get install python3-libnvinfer

进行trt模型转换，注意修改文件路径以及模型名称

/usr/src/tensorrt/bin/trtexec --onnx=results/model_initial_simplified.onnx --saveEngine=results/model_initial_simplified_fp16.engine --fp16





