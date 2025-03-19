# SCNN-and-Resnet50-on-Medical-MNIST-classification
This repository contains implementations of Spatial Convolutional Neural Network (SCNN) and ResNet50 architectures for classifying the Medical MNIST dataset. The Medical MNIST dataset is a collection of medical images that serve as a benchmark for evaluating image classification algorithms in the healthcare domain.
Overview
This repository contains implementations of Spatial Convolutional Neural Network (SCNN) and ResNet50 architectures for classifying the Medical MNIST dataset. The Medical MNIST dataset is a collection of medical images that serve as a benchmark for evaluating image classification algorithms in the healthcare domain.

Project Goals
The primary objectives of this project are:

To explore the effectiveness of SCNN and ResNet50 architectures in classifying medical images.
To compare the performance of both models in terms of accuracy, training time, and computational efficiency.
To provide a comprehensive framework for future research and development in medical image classification.
Dataset
The Medical MNIST dataset consists of various medical images, including:

X-rays
MRIs
CT scans
The dataset is structured similarly to the traditional MNIST dataset, making it easy to adapt existing models for medical image classification tasks.

Features
SCNN Implementation: A custom implementation of the Spatial Convolutional Neural Network tailored for medical image classification.
ResNet50 Implementation: Utilization of the pre-trained ResNet50 model for transfer learning, fine-tuned on the Medical MNIST dataset.
Data Preprocessing: Scripts for data loading, normalization, and augmentation to enhance model performance.
Training and Evaluation: Comprehensive training scripts with evaluation metrics to assess model performance.
Visualization: Tools for visualizing training progress, loss curves, and model predictions.
Getting Started
To get started with this project, follow these steps:

Clone the Repository:

BASH

git clone https://github.com/yourusername/medical-mnist-classification.git
cd medical-mnist-classification
Install Dependencies:
Ensure you have Python 3.x installed, then install the required packages:

BASH

pip install -r requirements.txt
Download the Dataset:
Download the Medical MNIST dataset and place it in the data/ directory.

Train the Models:
Run the training scripts for SCNN and ResNet50:

BASH

python train_scNN.py
python train_resnet50.py
Evaluate the Models:
Use the evaluation scripts to assess model performance:

BASH

python evaluate.py
Results
The results of the classification tasks, including accuracy and loss metrics, will be logged and can be visualized using the provided tools.

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Medical MNIST Dataset
Original ResNet Paper
SCNN Research Papers 
