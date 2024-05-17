# Qwen1.5 for depression classification 使用说明
## 代码准备
1. `git clone 本项目`
2. `cd Qwen-1.5-depressClassify`
## 环境
1. 自行安装conda
2. `conda create -n qwen python=3.10`
3. `conda activate qwen`
4. pip install -r requirements.txt
## 模型准备
在modelscope下载[Qwen1.5-7b-chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary)
下载方法具体见[本博客](https://www.daixd.xyz/huggingface-gfwed#559d8174dc2441d9a97c4b51468e211e)
记住本步骤下载模型文件存放的目录。
## 测试
1. `cd examples/sft`
2. 修改 inference_ori.py 文件中的模型文件路径，改成你下载的模型文件所在路径
3. 修改 inference_ori.py 文件中的模型文件路径，改成data里的output-860.jsonl(或者你有另外的测试数据就改成别的)
## 推理
运行cli_demo.py（同样需要修改模型路径）
