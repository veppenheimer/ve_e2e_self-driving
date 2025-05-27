端到端模型部署全流程
数据采集
推荐方法为利用手柄或者键盘遥控小车进行数据采集，数据尽量覆盖实战中小车全部的轨迹以便得到最稳定的表现。

可以数据采集可以使用项目中的data_collect.py进行，随后利用create_data_lists.py文件生成数据标签，用于train.py训练。

模型导出
为了得到在jetson nano板端更好的性能，采用将训练得到的pth模型导出为onnx模型、再转换为trt模型进行板端部署。

可在pc端利用export_onnx.py直接导出为onnx模型，随项目提供Onnx模型的检验、简化和推理文件用于评估Onnx模型的正确性和精度。

利用onnx生成trt引擎
将简化后的Onnx模型转换为trt模型需要在板端进行操作，首先在nano端安装模型转换所需要的库：

sudo apt-get install python3-libnvinfer

安装完成后即可开始trt模型转换，注意修改文件路径以及模型名称

/usr/src/tensorrt/bin/trtexec --onnx=results/veps_0414_simplified.onnx --saveEngine=results/veps_0414_simplified_fp16.engine --fp16

TRT模型推理
由于依赖库兼容问题，模型推理目前需要在python3.6版本下进行，这里给出使用conda创建虚拟环境并部署的流程，创建并激活虚拟环境：

conda create -n venv_name python=3.6

conda activate vnev_name

安装pycuda库
pip install pycuda==2019.1.2

随后可利用如下命令运行项目中附带的tensorRT_inference.py进行板端推理和验证，给出推理耗时。

LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python tensorRT_inference.py
注：若trt推理代码运行出现No Moduled Named tensorrt报错，可以通过将项目中附带的编译好的tensorrt包拷贝进当前虚拟环境下的lib库中解决，一般路径为:虚拟环境名称/lib/python3.6/site-packages
