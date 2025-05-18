# Emotion Detection using CNN on FER-2013 Dataset

This repository contains a PyTorch-based Convolutional Neural Network (CNN) model built to detect emotions from facial expressions using the **FER-2013** dataset.

---

## Project Overview

The model classifies facial images into **7 emotion classes**:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The architecture is a custom deep CNN with 5 convolutional layers followed by batch normalization, ReLU activations, max pooling, and fully connected layers with dropout. The model achieves approximately **65% accuracy** on the FER-2013 test set.

---

## Architecture

- Input: Grayscale images (1 channel)
- Conv Layers: 5 convolutional layers with increasing filter sizes (32 → 512)
- Batch Normalization and ReLU after each convolution
- MaxPooling layers for downsampling
- Fully connected layers: 256 → 512 → 7 (output classes)
- Dropout layer to reduce overfitting

---

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) is a publicly available dataset with 35,887 grayscale 48x48 pixel facial images labeled with 7 emotion categories.

---

## Installation

#### 1. Clone the repository
        git clone https://github.com/yourusername/emotion-detector.git
        cd emotion-detector

#### 3. Install required packages
        pip install -r requirements.txt

 