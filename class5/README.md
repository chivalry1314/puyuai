# 第五课 LMDeploy环境部署

## 创建开发机

打开InternStudio平台，创建开发机。填写开发机名称；选择镜像Cuda12.2-conda；选择10% A100*1GPU；点击“立即创建”。**注意请不要选择Cuda11.7-conda的镜像，新版本的lmdeploy会出现兼容性问题。**

## 创建conda环境

由于环境依赖项存在torch，下载过程可能比较缓慢。InternStudio上提供了快速创建conda环境的方法。打开命令行终端，创建一个名为`lmdeploy`的环境：
```bash
studio-conda -t lmdeploy -o pytorch-2.1.2
```

## 本地环境创建conda环境
注意，如果你在上一步已经在InternStudio开发机上创建了conda环境，这一步就没必要执行了。

打开命令行终端，让我们来创建一个名为lmdeploy的conda环境，python版本为3.10。

```bash
conda create -n lmdeploy -y python=3.10
```

## 安装LMDeploy

接下来，激活刚刚创建的虚拟环境。

```
conda activate lmdeploy
```

安装0.3.0版本的lmdeploy。
```
pip install lmdeploy[all]==0.3.0
```

等待安装结束就OK了！

## LMDeploy模型对话(chat)

### Huggingface与TurboMind

#### HuggingFace

**HuggingFace** 是一个高速发展的社区，包括Meta、Google、Microsoft、Amazon在内的超过5000家组织机构在为HuggingFace开源社区贡献代码、数据集和模型。可以认为是一个针对深度学习模型和数据集的在线托管社区，如果你有数据集或者模型想对外分享，网盘又不太方便，就不妨托管在HuggingFace。

托管在HuggingFace社区的模型通常采用HuggingFace格式存储，简写为**HF格式**。

但是HuggingFace社区的服务器在国外，国内访问不太方便。国内可以使用阿里巴巴的**MindScope**社区，或者上海AI Lab搭建的**OpenXLab**社区，上面托管的模型也通常采用**HF格式**。

#### TurboMind

TurboMind是LMDeploy团队开发的一款关于LLM推理的高效推理引擎，它的主要功能包括：LLaMa 结构模型的支持，continuous batch 推理模式和可扩展的 KV 缓存管理器。

TurboMind推理引擎仅支持推理TurboMind格式的模型。因此，TurboMind在推理HF格式的模型时，会首先自动将HF格式模型转换为TurboMind格式的模型。**该过程在新版本的LMDeploy中是自动进行的，无需用户操作。**

几个容易迷惑的点：

- TurboMind与LMDeploy的关系：LMDeploy是涵盖了LLM 任务全套轻量化、部署和服务解决方案的集成功能包，TurboMind是LMDeploy的一个推理引擎，是一个子模块。LMDeploy也可以使用pytorch作为推理引擎。
- TurboMind与TurboMind模型的关系：TurboMind是推理引擎的名字，TurboMind模型是一种模型存储格式，TurboMind引擎只能推理TurboMind格式的模型。

#### 下载模型

本次实战营已经在开发机的共享目录中准备好了常用的预训练模型，可以运行如下命令查看：

```bash
ls /root/share/new_models/Shanghai_AI_Laboratory/
```

显示如下，每一个文件夹都对应一个预训练模型。

以InternLM2-Chat-1.8B模型为例，从官方仓库下载模型。

#### InternStudio开发机上下载模型（推荐）

如果你是在InternStudio开发机上，可以按照如下步骤快速下载模型。

首先进入一个你想要存放模型的目录，本教程统一放置在Home目录。执行如下指令：

```bash
cd ~
```

然后执行如下指令由开发机的共享目录软链接或拷贝模型：

```bash
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
# cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
```

执行完如上指令后，可以运行“ls”命令。可以看到，当前目录下已经多了一个`internlm2-chat-1_8b`文件夹，即下载好的预训练模型。

```bash
ls
```

#### 由OpenXLab平台下载模型
注意，如果你在上一步已经从InternStudio开发机上下载了模型，这一步就没必要执行了。

上一步介绍的方法只适用于在InternStudio开发机上下载模型，如果是在自己电脑的开发环境上，也可以由OpenXLab下载。在开发机上还是建议通过拷贝的方式，因为从OpenXLab平台下载会占用大量带宽~

首先进入一个你想要存放模型的目录，本教程统一放置在Home目录。执行如下指令：

```bash
cd ~
```

OpenXLab平台支持通过Git协议下载模型。首先安装git-lfs组件。

- 对于root用于请执行如下指令：
  ```bash
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
  apt update
  apt install git-lfs   
  git lfs install  --system
  ```
- 对于非root用户需要加sudo，请执行如下指令：
  ```bash
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
  sudo apt update
  sudo apt install git-lfs   
  sudo git lfs install  --system
  ```
安装好git-lfs组件后，由OpenXLab平台下载InternLM2-Chat-1.8B模型：
```bash
git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-1.8b.git
```
这一步骤可能耗时较长，主要取决于网速，耐心等待即可。

下载完成后，运行`ls`指令，可以看到当前目录下多了一个`internlm2-chat-1.8b`文件夹，即下载好的预训练模型。

```bash
ls
```

注意！从OpenXLab平台下载的模型文件夹命名为`1.8b`，而从InternStudio开发机直接拷贝的模型文件夹名称是`1_8b`，为了后续文档统一，这里统一命名为`1_8b`。

```bash
mv /root/internlm2-chat-1.8b /root/internlm2-chat-1_8b
```

#### 使用Transformer库运行模型
Transformer库是Huggingface社区推出的用于运行HF模型的官方库。

在2.2中，我们已经下载好了InternLM2-Chat-1.8B的HF模型。下面我们先用Transformer来直接运行InternLM2-Chat-1.8B模型，后面对比一下LMDeploy的使用感受。

现在，让我们点击左上角的图标，打开VSCode。

![2.3_2](images/2.3_2.jpg)

在左边栏空白区域单击鼠标右键，点击`Open in Intergrated Terminal`。

![2.3_3](images/2.3_3.jpg)

等待片刻，打开终端。

![2.3_4](images/2.3_4.jpg)

在终端中输入如下指令，新建`pipeline_transformer.py`。

```bash
touch /root/pipeline_transformer.py
```

回车执行指令，可以看到侧边栏多出了`pipeline_transformer.py`文件，点击打开。后文中如果要创建其他新文件，也是采取类似的操作。

![2.3_5](images/2.3_5.jpg)

将以下内容复制粘贴进入pipeline_transformer.py。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)
```

![2.3_6](images/2.3_6.jpg)

按`Ctrl+S`键保存（Mac用户按`Command+S`）。

回到终端，激活conda环境。

```bash
conda activate lmdeploy
```

运行python代码：

```bash
python /root/pipeline_transformer.py
```

得到输出：

![transformer库运行模型结果](images/transformer_result.png)

记住这种感觉，一会儿体验一下LMDeploy的推理速度，感受一下对比~（手动狗头）

#### 使用LMDeploy与模型对话
这一小节我们来介绍如何应用LMDeploy直接与模型进行对话。

首先激活创建好的conda环境：
```
conda activate lmdeploy
```
使用LMDeploy与模型进行对话的通用命令格式为：
```
lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]
```
例如，您可以执行如下命令运行下载的1.8B模型：
```
lmdeploy chat /root/internlm2-chat-1_8b
```

比如输入“给我讲个笑话”，然后按两下回车键。

![tell_joke](images/tell_joke.png)

速度是不是明显比原生Transformer快呢~当然，这种感受可能不太直观，感兴趣的佬可以查看拓展部分“6.3 定量比较LMDeploy与Transformer库的推理速度”。

输入“exit”并按两下回车，可以退出对话。

**拓展内容**：有关LMDeploy的chat功能的更多参数可通过-h命令查看。

```
lmdeploy chat -h
```

### LMDeploy模型量化(lite)

本部分内容主要介绍如何对模型进行量化。主要包括 KV8量化和W4A16量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。

正式介绍 LMDeploy 量化方案前，需要先介绍两个概念：

- 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速度。
- 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。

那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用KV8量化和W4A16量化。KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

#### 设置最大KV Cache缓存大小

KV Cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。在大规模训练和推理中，KV Cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，KV Cache全部存储于显存，以加快访存速度。当显存空间不足时，也可以将KV Cache放在内存，通过缓存管理器控制将当前需要使用的数据放入显存。

模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、KV Cache占用的显存，以及中间运算结果占用的显存。LMDeploy的KV Cache管理器可以通过设置 **--cache-max-entry-count** 参数，控制KV缓存占用剩余显存的最大比例。默认的比例为0.8。

下面通过几个例子，来看一下调整 **--cache-max-entry-count** 参数的效果。首先保持不加该参数（默认0.8），运行1.8B模型。

```bash
lmdeploy chat /root/internlm2-chat-1_8b
```

与模型对话，查看右上角资源监视器中的显存占用情况。

![cache_max_08](images/cache_max_08.png)

此时显存占用为7856MB。下面，改变 **--cache-max-entry-count** 参数，设为0.5。

```bash
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.5
```

与模型对话，再次查看右上角资源监视器中的显存占用情况。

![cache_max_05](images/cache_max_05.png)

看到显存占用明显降低，变为6600M。

下面来一波“极限”，把 **--cache-max-entry-count** 参数设置为0.01，约等于禁止KV Cache占用显存。
```bash
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.01
```
然后与模型对话，可以看到，此时显存占用仅为4096MB，代价是会降低模型推理速度。
![cache_max_001](images/cache_max_001.png)

#### 使用W4A16量化
LMDeploy使用AWQ算法，实现模型4bit权重量化。推理引擎TurboMind提供了非常高效的4bit推理cuda kernel，性能是FP16的2.4倍以上。它支持以下NVIDIA显卡：
- 图灵架构（sm75）：20系列、T4
- 安培架构（sm80,sm86）：30系列、A10、A16、A30、A100
- Ada Lovelace架构（sm90）：40 系列

运行前，首先安装一个依赖库。
```bash
pip install einops==0.7.0
```

仅需执行一条命令，就可以完成模型量化工作。

```
lmdeploy lite auto_awq \
   /root/internlm2-chat-1_8b \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/internlm2-chat-1_8b-4bit
```
运行时间较长，请耐心等待。量化工作结束后，新的HF模型被保存到internlm2-chat-1_8b-4bit目录。下面使用Chat功能运行W4A16量化后的模型。
```bash
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq
```
为了更加明显体会到W4A16的作用，我们将KV Cache比例再次调为0.01，查看显存占用情况。
```bash
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.01
```
可以看到，显存占用变为2350MB，明显降低。

![w4a16_cache001](images/w4a16_cache001.png)

**拓展内容**：有关LMDeploy的lite功能的更多参数可通过-h命令查看。

```bash
lmdeploy lite -h
```
### LMDeploy服务(serve)
在第二章和第三章，我们都是在本地直接推理大模型，这种方式成为本地部署。在生产环境下，我们有时会将大模型封装为API接口服务，供客户端访问。

我们来看下面一张架构图：

![4_1](images/4_1.jpg)

我们把从架构上把整个服务流程分成下面几个模块。

- 模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
- API Server。中间协议层，把后端推理/服务通过HTTP，gRPC或其他形式的接口，供前端调用。
- Client。可以理解为前端，与用户交互的地方。通过通过网页端/命令行去调用API接口，获取模型推理/服务。

值得说明的是，以上的划分是一个相对完整的模型，但在实际中这并不是绝对的。比如可以把“模型推理”和“API Server”合并，有的甚至是三个流程打包在一起提供服务。

#### 启动API服务器

通过以下命令启动API服务器，推理`internlm2-chat-1_8b`模型：

```bash
lmdeploy serve api_server \
    /root/internlm2-chat-1_8b \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```

其中，model-format、quant-policy这些参数是与第三章中量化推理模型一致的；server-name和server-port表示API服务器的服务IP与服务端口；tp参数表示并行数量（GPU数量）。

通过运行以上指令，我们成功启动了API服务器，请勿关闭该窗口，后面我们要新建客户端连接该服务。

可以通过运行一下指令，查看更多参数及使用方法：

```bash
lmdeploy serve api_server -h
```

你也可以直接打开```http://{host}:23333```查看接口的具体使用说明，如下图所示。

注意，这一步由于Server在远程服务器上，所以本地需要做一下ssh转发才能直接访问。`在你本地打开一个cmd窗口`，输入命令如下：

```bash
ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的ssh端口号
```

#### 命令行客户端连接API服务器
本节中，我们要新建一个命令行客户端去连接API服务器。首先通过VS Code新建一个终端：
![4.2_1](images/4.2_1.jpg)

激活conda环境。

```bash
conda activate lmdeploy
```
运行命令行客户端：
```bash
lmdeploy serve api_client http://localhost:23333
```
运行后，可以通过命令行窗口直接与模型对话：
![lmdeploy_client](images/lmdeploy_client.png)

现在你使用的架构是这样的：
![4.2_4](images/4.2_4.jpg)

#### 网页客户端连接API服务器
关闭刚刚的VSCode终端，但服务器端的终端不要关闭。

新建一个VSCode终端，激活conda环境。

```bash
conda activate lmdeploy
```
使用Gradio作为前端，启动网页客户端。
```bash
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

运行命令后，网页客户端启动。在电脑本地新建一个cmd终端，新开一个转发端口：

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的ssh端口号>
```

打开浏览器，访问地址`http://127.0.0.1:6006`

然后就可以与模型进行对话了！
![web_client](images/web_client.png)

现在你使用的架构是这样的：

![4.3_3](images/4.3_3.png)

#### Python代码集成
在开发项目时，有时我们需要将大模型推理集成到Python代码里面。

##### Python代码集成运行1.8B模型
首先激活conda环境。
```
conda activate lmdeploy
```
新建Python源代码文件`pipeline.py`。
```
touch /root/pipeline.py
```

打开`pipeline.py`，填入以下内容。
```python
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

>代码解读：\
>- 第1行，引入lmdeploy的pipeline模块 \
>- 第3行，从目录“./internlm2-chat-1_8b”加载HF模型 \
>- 第4行，运行pipeline，这里采用了批处理的方式，用一个列表包含两个输入，lmdeploy同时推理两个输入，产生两个输出结果，结果返回给response \
>- 第5行，输出response

保存后运行代码文件：

```bash
python /root/pipeline.py
```

![python_integrated](images/python_integrated.png)

##### 向TurboMind后端传递参数
在第3章，我们通过向lmdeploy传递附加参数，实现模型的量化推理，及设置KV Cache最大占用比例。在Python代码中，可以通过创建TurbomindEngineConfig，向lmdeploy传递参数。

以设置KV Cache占用比例为例，新建python文件`pipeline_kv.py`。

```bash
touch /root/pipeline_kv.py
```

打开`pipeline_kv.py`，填入如下内容：

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占比调整为总显存的 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('/root/internlm2-chat-1_8b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

保存后运行python代码：

```bash
python /root/pipeline_kv.py
```
![pipline_kv_result](images/pipline_kv_result.png)

## 拓展部分

### 使用LMDeploy运行视觉多模态大模型llava

最新版本的LMDeploy支持了llava多模态模型，下面演示使用pipeline推理llava-v1.6-7b。注意，运行本pipeline最低需要30%的InternStudio开发机，请完成基础作业后向助教申请权限。

首先激活conda环境。

```bash
conda activate lmdeploy
```

安装llava依赖库。

```bash
pip install git+https://github.com/haotian-liu/LLaVA.git@4e2277a060da264c4f21b364c867cc622c945874
```

新建一个python文件，比如`pipeline_llava.py`。
```bash
touch /root/pipeline_llava.py
```

打开`pipeline_llava.py`，填入内容如下：
```python
from lmdeploy.vl import load_image
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

>代码解读： \
>- 第1行引入用于载入图片的load_image函数，第2行引入了lmdeploy的pipeline模块， \
>- 第5行创建了pipeline实例 \
>- 第7行从github下载了一张关于老虎的图片，如下：
![tiger](images/tiger.png) \
>- 第8行运行pipeline，输入提示词“describe this image”，和图片，结果返回至response \
>- 第9行输出response

保存后运行pipeline。

```bash
python /root/pipeline_llava.py
```

运行后的输出结果为：

![tiger_desc_result](images/tiger_desc_result.png)

我们也可以通过Gradio来运行llava模型。新建python文件gradio_llava.py。

```bash
touch /root/gradio_llava.py
```

打开文件，填入以下内容：
```python
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()   
```

运行python程序。

```bash
python /root/gradio_llava.py
```

通过ssh转发一下7860端口。

```
ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p <你的ssh端口>
```

通过浏览器访问http://127.0.0.1:7860。

然后就可以使用啦~

效果如下所示：

![baby_desc_result](images/baby_desc_result.png)

### 使用LMDeploy运行第三方大模型

LMDeploy不仅支持运行InternLM系列大模型，还支持其他第三方大模型。支持的模型列表如下：

|       Model        |    Size    |
| :----------------: | :--------: |
|       Llama        |  7B - 65B  |
|       Llama2       |  7B - 70B  |
|      InternLM      |  7B - 20B  |
|     InternLM2      |  7B - 20B  |
| InternLM-XComposer |     7B     |
|        QWen        |  7B - 72B  |
|      QWen-VL       |     7B     |
|      QWen1.5       | 0.5B - 72B |
|    QWen1.5-MoE     |   A2.7B    |
|      Baichuan      |  7B - 13B  |
|     Baichuan2      |  7B - 13B  |
|     Code Llama     |  7B - 34B  |
|      ChatGLM2      |     6B     |
|       Falcon       | 7B - 180B  |
|         YI         |  6B - 34B  |
|      Mistral       |     7B     |
|    DeepSeek-MoE    |    16B     |
|    DeepSeek-VL     |     7B     |
|      Mixtral       |    8x7B    |
|       Gemma        |   2B-7B    |
|        Dbrx        |    132B    |

可以从Modelscope，OpenXLab下载相应的HF模型，下载好HF模型，下面的步骤就和使用LMDeploy运行InternLM2一样啦~

## 6.3 定量比较LMDeploy与Transformer库的推理速度差异

为了直观感受LMDeploy与Transformer库推理速度的差异，让我们来编写一个速度测试脚本。测试环境是30%的InternStudio开发机。

先来测试一波Transformer库推理Internlm2-chat-1.8b的速度，新建python文件，命名为`benchmark_transformer.py`，填入以下内容：

```py
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response, history = model.chat(tokenizer, inp, history=[])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response, history = model.chat(tokenizer, inp, history=history)
    total_words += len(response)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```

运行python脚本：

```sh
python benchmark_transformer.py
```

得到运行结果：

![benchmark_transformer](images/benchmark_transformer.png)

可以看到，Transformer库的推理速度约为27.702 words/s，注意单位是words/s，不是token/s，word和token在数量上可以近似认为成线性关系。

下面来测试一下LMDeploy的推理速度，新建python文件`benchmark_lmdeploy.py`，填入以下内容：

```py
import datetime
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response = pipe([inp])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response = pipe([inp])
    total_words += len(response[0].text)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```

运行脚本：

```sh
python benchmark_lmdeploy.py
```

得到运行结果：
![benchmark_lmdeploy](images/benchmark_lmdeploy.png)

可以看到，LMDeploy的推理速度约为222.563 words/s，是Transformer库的8倍。


## 进阶作业

### 作业1

设置KV Cache最大占用比例为0.4，开启W4A16量化，以命令行方式与模型对话。

运行以下命令，开启W4A16量化：
```bash
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4
```
![w4a16量化04](images/w4a16量化04.png)

对话的结果如下图所示：

![w4a16量化04对话](images/w4a16量化04对话.png)

### 作业2

以API Server方式启动 lmdeploy，开启 W4A16量化，调整KV Cache的占用比例为0.4，分别使用命令行客户端与Gradio网页客户端与模型对话。

运行以下命令，开启W4A16量化的API Server：

```bash
lmdeploy serve api_server \
    /root/internlm2-chat-1_8b-4bit \
    --model-format awq \
	--cache-max-entry-count 0.4 \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
运行后，效果图如下所示：
![W4A16量化的API_Server](images/W4A16量化的API_Server.png)

激活conda环境。

```bash
conda activate lmdeploy
```
运行命令行客户端：
```bash
lmdeploy serve api_client http://localhost:23333
```
运行后，可以通过命令行窗口直接与模型对话：

![W4A16量化的API_Server console](images/W4A16量化的API_Server_console.png)

新建一个VSCode终端，激活conda环境。

```bash
conda activate lmdeploy
```
使用Gradio作为前端，启动网页客户端。
```bash
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

运行命令后，网页客户端启动。在电脑本地新建一个cmd终端，新开一个转发端口：

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的ssh端口号>
```

打开浏览器，访问地址`http://127.0.0.1:6006`

然后就可以与模型进行对话了！
![web_client](images/web_client.png)

### 作业3
使用W4A16量化，调整KV Cache的占用比例为0.4，使用Python代码集成的方式运行internlm2-chat-1.8b模型。

打开`pipeline_kv.py`，填入如下内容：

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占比调整为总显存的 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.4)

pipe = pipeline('/root/internlm2-chat-1_8b-4bit',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```

保存后运行python代码：

```bash
python /root/pipeline_kv.py
```
![work3_result](images/work3_result.png)