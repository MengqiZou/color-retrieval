# 项目安装与运行指南  
  
## 1. 环境准备  
  
为了运行本项目，您需要先安装Anaconda或Miniconda，这两个都是流行的Python数据科学发行版，包含了conda包管理器和Python。  
  
如果您还没有安装Anaconda或Miniconda，请访问[Anaconda官网](https://www.anaconda.com/download/)进行下载和安装。  
  
## 2. 创建虚拟环境  
  
打开命令行工具（如Terminal、Command Prompt或PowerShell），然后运行以下命令来创建一个名为`myenv`的新conda环境：  
  
```bash  
conda create --name myenv
```

## 3. 初始化conda
为了能够在您的shell中方便地激活和使用conda环境，您需要运行以下命令来初始化conda：

```bash  
conda init
```
## 4. 激活虚拟环境
在创建并初始化了conda环境之后，您可以使用以下命令来激活它：

```bash  
conda activate myenv
```

## 5. 安装依赖包
本项目的依赖项都列在requirements.txt文件中。在激活了myenv环境后，您可以使用pip来安装这些依赖项：

```bash  
pip install -r requirements.txt
```
## 6. 模型checkpoints

### 创建checkpoints文件夹

```bash 
mkdir colorDetectionFastApi/checkpoints
```

### 安装五官detector checkpoint -- face_landmarker_v2_with_blendshapes.task:

```bash  
wget -O colorDetectionFastApi/checkpoints/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
### 安装hair segmentor checkpoint -- hair_segmenter.tflite: 

```bash  
wget -O colorDetectionFastApi/checkpoints/hair_segmenter.tflite -q https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
```
### 两个checkpint都成功下载成功后的文件结构式：

```shell
├── README.md
├── colorDetectionFastApi
│   ├── __init__.py
│   ├── checkpoints
│   │   ├── face_landmarker_v2_with_blendshapes.task
│   │   └── hair_segmenter.tflite
│   ├── detectors
│   │   ├── __init__.py
│   │   └── mediapipe.py
│   ├── main.py
│   ├── routers
│   │   ├── __init__.py
│   │   └── mediapipe.py
│   └── schemas
│       └── mediapipe.py
├── outputs
├── poetry.lock
├── pyproject.toml
├── requirements.txt
```
## 7. 运行项目
当所有依赖项都安装完毕后，您可以使用以下命令来运行项目的主程序：

```bash  
python /root/autodl-tmp/color-retrieval/colorDetectionFastApi/main.py
```
