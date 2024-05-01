# Llama3-Tutorial（Llama 3 超级课堂）
## 第一节 Llama 3 本地 Web Demo 部署
### 环境配置
注意：镜像一定要用Cuda11.7-conda版本
```
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### 模型下载
安装 git-lfs 依赖
```
conda install git-lfs
git-lfs install
```
下载模型 （InternStudio 中不建议执行这一步）
```
mkdir -p ~/model
cd ~/model
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```
或者软链接 InternStudio 中的模型
```
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct ~/model/Meta-Llama-3-8B-Instruct
```
### Web Demo部署
```
cd ~
git clone https://github.com/SmartFlowAI/Llama3-XTuner-CN
```
安装 XTuner 时会自动安装其他依赖
```
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```
运行 web_demo.py
```
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  ~/model/Meta-Llama-3-8B-Instruct
```
![Llama3界面](images/model.png)

## 第二节 XTuner 微调 Llama3 个人小助手认知
### 环境配置
### 下载模型
### Web Demo 部署
环境配置、下载模型、Web Demo 部署见第一节

### 自我认知训练数据集准备
```
cd ~/Llama3-XTuner-CN
python tools/gdata.py 
```
以上脚本在生成了 ~/Llama3-XTuner-CN/data/personal_assistant.json 数据文件格式如下所示：
```
[
    {
        "conversation": [
            {
                "system": "你是一个懂中文的小助手",
                "input": "你是（请用中文回答）",
                "output": "您好，我是SmartFlowAI，一个由 SmartFlowAI 打造的人工智能助手，请问有什么可以帮助您的吗？"
            }
        ]
    },
    {
        "conversation": [
            {
                "system": "你是一个懂中文的小助手",
                "input": "你是（请用中文回答）",
                "output": "您好，我是SmartFlowAI，一个由 SmartFlowAI 打造的人工智能助手，请问有什么可以帮助您的吗？"
            }
        ]
    }
]
```
### XTuner配置文件准备
小编为大佬们修改好了configs/assistant/llama3_8b_instruct_qlora_assistant.py 配置文件(主要修改了模型路径和对话模板)请直接享用～
### 训练模型
```
cd ~/Llama3-XTuner-CN

# 开始训练,使用 deepspeed 加速，A100 40G显存 耗时24分钟
xtuner train configs/assistant/llama3_8b_instruct_qlora_assistant.py --work-dir /root/llama3_pth

# Adapter PTH 转 HF 格式
xtuner convert pth_to_hf /root/llama3_pth/llama3_8b_instruct_qlora_assistant.py \
  /root/llama3_pth/iter_500.pth \
  /root/llama3_hf_adapter

# 模型合并
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/model/Meta-Llama-3-8B-Instruct \
  /root/llama3_hf_adapter\
  /root/llama3_hf_merged
```
### 推理验证
```
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  /root/llama3_hf_merged
```
![Llama3界面](images/modelafterft.png)

## 第三节 XTuner 微调 Llama3 图片理解多模态
随着 XTuner 团队放出了基于 Llama3-8B 的 LLaVA 模型，我们也是第一时间与 XTuner 团队取得了联系，并获得了他们已经预训练好的 Image Projector。接下来，我们将带大家基于 Llama3-8B-Instruct 和 XTuner 团队预训练好的 Image Projector 微调自己的多模态图文理解模型 LLaVA。
### 环境配置
### 下载模型
### Web Demo 部署
环境配置、下载模型、Web Demo 部署见第一节

如果在前面的课程中已经配置好了环境，在这里也可以选择直接执行 conda activate llama3 以进入环境。

最后我们 clone 本教程仓库。
```
cd ~
git clone https://github.com/SmartFlowAI/Llama3-Tutorial
```
### 模型准备
#### 准备 Llama3 权重
在微调开始前，我们首先来准备 Llama3-8B-Instruct 模型权重。
##### InternStudio
```
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct .
```
##### 非 InternStudio
我们选择从 OpenXLab 上下载 Meta-Llama-3-8B-Instruct 的权重。
```
mkdir -p ~/model
cd ~/model
git lfs install
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```
#### 准备 Visual Encoder 权重
我们接下来准备 Llava 所需要的 openai/clip-vit-large-patch14-336，权重，即 Visual Encoder 权重。
##### InternStudio
```
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/openai/clip-vit-large-patch14-336 .
```
##### 非 InternStudio
可以访问 https://huggingface.co/openai/clip-vit-large-patch14-336 以进行下载。
#### 准备 Image Projector 权重
然后我们准备 Llava 将要用到的 Image Projector 部分权重
##### InternStudio
```
mkdir -p ~/model
cd ~/model
ln -s /root/share/new_models/xtuner/llama3-llava-iter_2181.pth .
```
##### 非 InternStudio
相关权重可以访问：https://huggingface.co/xtuner/llava-llama-3-8b 以及 https://huggingface.co/xtuner/llava-llama-3-8b-v1_1 。（已经过微调，并非 Pretrain 阶段的 Image Projector）

#### 数据准备
我们按照 https://github.com/InternLM/Tutorial/blob/camp2/xtuner/llava/xtuner_llava.md 中的教程来准备微调数据。为了让大家可以快速上手，我们选择了使用过拟合的方式快速实现。
可以执行以下代码：
```
cd ~
git clone https://github.com/InternLM/tutorial -b camp2
python ~/tutorial/xtuner/llava/llava_data/repeat.py \
  -i ~/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o ~/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
```
### 微调过程
#### 训练启动
我们已经为大家准备好了可以一键启动的配置文件，主要是修改好了模型路径、对话模板以及数据路径。

我们使用如下指令以启动训练：
```
xtuner train ~/Llama3-Tutorial/configs/llama3-llava/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py --work-dir ~/llama3_llava_pth --deepspeed deepspeed_zero2
```