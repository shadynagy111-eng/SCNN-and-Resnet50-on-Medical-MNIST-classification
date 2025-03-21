# Spiking Convolutional Neural Network and ResNet-50 on Medical MNIST Classification

## Overview

This repository contains implementations of Spiking Convolutional Neural Network (SCNN) and ResNet-50 architectures for classifying the Medical MNIST dataset. The Medical MNIST dataset is a collection of medical images that serve as a benchmark for evaluating image classification algorithms in the healthcare domain. 🏥📊

## Project Goals

The primary objectives of this project are:

- To explore the effectiveness of SCNN and ResNet-50 architectures in classifying medical images. 🧠
- To compare the performance of both models in terms of accuracy, training time, and computational efficiency. ⚖️
- To provide a comprehensive framework for future research and development in medical image classification. 🔬

## Dataset

The Medical MNIST dataset consists of various medical images, including:

- X-rays 🩻
- MRIs 🧲
- CT scans 🖥️

The dataset is structured similarly to the traditional MNIST dataset, making it easy to adapt existing models for medical image classification tasks.

## Features

- **SCNN Implementation**: A custom implementation of the Spiking Convolutional Neural Network tailored for medical image classification. 🧠
- **ResNet-50 Implementation**: Utilization of ResNet-50 model on the Medical MNIST dataset. 🖼️
- **Data Preprocessing**: Scripts for data loading, normalization, and augmentation to enhance model performance. 📈
- **Training and Evaluation**: Comprehensive training scripts with evaluation metrics to assess model performance. 🏋️‍♂️
- **Visualization**: Tools for visualizing training progress, loss curves, and model predictions. 📊

## File Structure

```
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── scnn.py
│   ├── resnet50.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── README.md
├── LICENSE
└── requirements.txt
```

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/SCNN-and-Resnet50-on-Medical-MNIST-classification.git
    cd SCNN-and-Resnet50-on-Medical-MNIST-classification
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Preprocess the data**:
    ```bash
    python scripts/preprocess.py
    ```

4. **Train the models**:
    ```bash
    python scripts/train.py --model scnn
    python scripts/train.py --model resnet50
    ```

5. **Evaluate the models**:
    ```bash
    python scripts/evaluate.py --model scnn
    python scripts/evaluate.py --model resnet50
    ```

## Diagrams

### Model Architecture

#### SCNN Architecture
![SCNN Architecture](path/to/scnn_architecture.png)

#### ResNet-50 Architecture
![ResNet50 Architecture](path/to/resnet50_architecture.png)

### Training Process
![Training Process](path/to/training_process.png)

## Tables

### Performance Comparison

| Model   | Accuracy | Training Time | Computational Efficiency |
|---------|----------|---------------|--------------------------|
| SCNN    | ____     | _______       | ____                     |
| ResNet-50 | ____     | _______       | ____                     |

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request. 🙌

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📄

## Acknowledgments

- [Medical MNIST Dataset](https://medmnist.com/) 🏥
- [Original ResNet Paper](https://arxiv.org/abs/1512.03385) 📄
- SCNN Research Papers 📚
