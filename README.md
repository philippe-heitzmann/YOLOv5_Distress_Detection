## Project Overview

- This project focuses on building deep learning model training and evaluation pipelines for object detection of pavement road distresses in response to [IEEE 2020 Global Road Detection Challenge](https://rdd2020.sekilab.global/)
- Object detection models used: YOLO, Faster R-CNN
- Computer Vision frameworks used: Pytorch, Tensorflow

## Paper

- See [paper](https://arxiv.org/abs/2202.13285) for final research insights & results

## GPU setup for training models

Hardware used: Nvidia RTX 3090
* Given the RTX 3090 embeds Ampere architecture, it will only work with Nvidia Driver 450+ versions only.
![image](https://user-images.githubusercontent.com/8759492/139563640-602d5dfb-48dc-44fa-b911-d9cbd0404671.png)
https://docs.nvidia.com/deploy/cuda-compatibility/index.html
* Given we can only work with Nvidia Driver versions 450+, we will require CUDA versions 11.0+ 
![image](https://user-images.githubusercontent.com/8759492/139563706-be335ccb-c0e1-41cf-94b7-27c541c0adc1.png)
https://docs.nvidia.com/deploy/cuda-compatibility/index.html
* Given we can now only work with CUDA versions 11.0+, we will require cuDNN versions 8/0+ 
https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html


- See https://medium.com/@dun.chwong/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc for more details 

## Data pipeline structure 

1. XML_to_TXT_Annotation_Conversion_Pipeline.ipynb to convert XML annotation files to TXT for YOLOv5 use 
1. A01 - Load and Augment an Image.ipynb to define augmentations to apply to input images