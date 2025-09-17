# YOLOv5 Distress Detection

A deep learning project for automated detection of pavement road distresses using computer vision techniques, developed in response to the [IEEE 2020 Global Road Detection Challenge](https://rdd2020.sekilab.global/).

## Overview

This project implements object detection models to identify and classify various types of road distresses from images. The solution uses state-of-the-art deep learning frameworks and models to achieve accurate detection results.

### Key Features

- **Object Detection Models**: YOLOv5, Faster R-CNN
- **Frameworks**: PyTorch, TensorFlow
- **Target Application**: Road distress detection and classification
- **Research Paper**: [View published results](https://arxiv.org/abs/2202.13285)

## Project Structure

```
├── data_preprocessing/          # Data preprocessing utilities
├── notebooks/                   # Jupyter notebooks for analysis and training
│   ├── modeling/               # Model training notebooks
│   ├── adhoc/                  # Ad-hoc analysis notebooks
│   └── titanmu/                # TITANMU model experiments
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended: NVIDIA RTX 3090)
- Docker (for containerized training)

### Hardware Requirements

**Recommended Hardware**: NVIDIA RTX 3090

**Driver and CUDA Requirements**:
- NVIDIA Driver: 450+ (required for Ampere architecture)
- CUDA: 11.0+ (compatible with driver 450+)
- cuDNN: 8.0+ (compatible with CUDA 11.0+)

For detailed setup instructions, see: [RTX 3090 Deep Learning Setup Guide](https://medium.com/@dun.chwong/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc)

### Docker Setup

This project uses Docker for consistent training environments. The notebooks reference YOLOv5 Docker images for model training.

#### Running with Docker

1. **Pull the YOLOv5 Docker image**:
   ```bash
   docker pull ultralytics/yolov5:latest
   ```

2. **Run the container with GPU support**:
   ```bash
   docker run --gpus all -it --rm -v $(pwd):/workspace ultralytics/yolov5:latest
   ```

3. **For interactive development**:
   ```bash
   docker run --gpus all -it --rm -v $(pwd):/workspace -p 8888:8888 ultralytics/yolov5:latest jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
   ```

4. **Access Jupyter Lab**:
   Open your browser and navigate to `http://localhost:8888`

#### Docker Compose (Alternative)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'
services:
  yolov5:
    image: ultralytics/yolov5:latest
    container_name: yolov5-distress-detection
    runtime: nvidia
    volumes:
      - .:/workspace
    ports:
      - "8888:8888"
    command: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

Then run:
```bash
docker-compose up
```

### Data Pipeline

The project includes a structured data processing pipeline:

1. **XML to TXT Conversion**: `XML_to_TXT_Annotation_Conversion_Pipeline.ipynb`
   - Converts XML annotation files to TXT format for YOLOv5 compatibility

2. **Image Augmentation**: `A01 - Load and Augment an Image.ipynb`
   - Defines and applies data augmentation techniques to training images

## Usage

1. **Data Preparation**: Run the XML to TXT conversion notebook to prepare your annotations
2. **Model Training**: Use the notebooks in the `modeling/` directory to train YOLOv5 models
3. **Evaluation**: Evaluate model performance using the prediction notebooks

## Research

This project is based on research published in the paper available at: [arXiv:2202.13285](https://arxiv.org/abs/2202.13285)

## Contributing

Please refer to the project objectives in `objectives.md` for current goals and development priorities.