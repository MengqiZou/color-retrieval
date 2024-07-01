# Hosting YOLOv8 With FastAPI

## Introduction
In the ever-evolving landscape of computer vision and machine learning, two powerful technologies have emerged as key players in their respective domains: YOLO (You Only Look Once) and FastAPI. YOLO has gained fame for its real-time object detection capabilities, while FastAPI has earned a reputation as one of the most efficient and user-friendly web frameworks for building APIs. In this blog post, we'll explore the exciting synergy that can be achieved by hosting YOLOv8, a state-of-the-art YOLO variant, with FastAPI.

![FastAPI And YOLO](./images/top-view.png)

First, let's briefly introduce FastAPI. FastAPI is a Python web framework that simplifies the development of APIs with incredible speed and ease. It is designed for high-performance and productivity, offering automatic generation of interactive documentation and type hints, which are a boon for developers. With FastAPI, you can build robust APIs quickly, making it an ideal choice for integrating machine learning models, like YOLOv8, into web applications.

On the other side of the equation is YOLO, a groundbreaking object detection model that has become a cornerstone of computer vision applications. YOLO excels at identifying objects in images and video streams in real-time. YOLOv8 is the latest iteration, bringing even more accuracy and speed to the table. Combining the power of YOLOv8 with the efficiency of FastAPI opens up exciting possibilities for building interactive and efficient object detection applications.

In this blog post, we will dive into the process of hosting YOLOv8 with FastAPI, demonstrating how to create a web-based API that can analyze images. By the end of this guide, you'll have a solid understanding of how to leverage these two technologies to build your own object detection applications, whether it's for security, surveillance, or any other use case that demands accurate and speedy object detection. Let's embark on this journey of integrating the cutting-edge YOLOv8 with the versatile FastAPI framework for a truly powerful and responsive object detection experience.

## Directory Structure
First, I do always like to split my code across multiple files. In
my opinion, it just makes it easier to read for me. I would be doing
a disservice if I didn't accurately show you the structure layout
so you can understand the imports that are happening between files:

```shell
|____yolofastapi
| |____routers
| | |______init__.py
| | |____yolo.py
| |______init__.py
| |____schemas
| | |____yolo.py
| |____detectors
| | |______init__.py
| | |____yolov8.py
| |____main.py
```

At the top level, we have the `yolofastapi` directory which will be our
python application. Within there, there are a few directories:

1. `routers` - The endpoints / REST routers that our application will expose.
               If, for example, you wanted to add a new `GET` endpoint, you
               could add that in this directory.
2. `schemas` - This directory will show our request/response schemas that our
               routers will either expect or return. Pydantic makes the 
               serialization of these objects a breeze!
3. `detectors` - This is the fun stuff! We will put our `yolo` or other detection
                 models/wrappers in this directory. In our example, we will only
                 using `yolov8n`, but you could extend this to other detectors
                 or yolo versions as well.

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

## 6. 运行项目
当所有依赖项都安装完毕后，您可以使用以下命令来运行项目的主程序：

```bash  
pip install -r requirements.txt
```