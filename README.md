# Emotion Detection using CNN on FER-2013 Dataset

This repository contains a PyTorch-based Convolutional Neural Network (CNN) model built to detect emotions from facial expressions using the **FER-2013** dataset.

## Project Structure 

<pre> emotion-detector/ │ ├── CustomDataset/ │ ├── train/ # Training images organized by emotion labels │ └── test/ # Test images organized by emotion labels │ ├── models/ │ └── modeldrop4.pth # Trained PyTorch model (not committed to Git) │ ├── helper_functions.py # Utility functions for preprocessing, evaluation, etc. ├── ImageClassifier.py # Contains the CustomCNN model architecture ├── trainNN.py # Script to train the CNN on FER-2013 dataset ├── live.py # Live emotion detection using webcam and OpenCV ├── live.ipynb # Jupyter notebook version of live detection ├── main.ipynb # Notebook for model training/visualization/testing ├── requirements.txt # Python package dependencies ├── README.md # Project documentation └── .gitignore # Files and folders ignored by Git </pre>


## 🚀 Project Overview

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

## 🧠 Model Architecture Diagrammatically

The model is a custom 5-layer convolutional neural network with Batch Normalization and Dropout:

```python
Conv2d → BatchNorm → ReLU → MaxPool  
   ↓  
Conv2d → BatchNorm → ReLU → MaxPool  
   ↓  
Conv2d → BatchNorm → ReLU  
   ↓  
Conv2d → BatchNorm → ReLU → MaxPool  
   ↓  
Conv2d → BatchNorm → ReLU → MaxPool  
   ↓  
Flatten → Dense (256) → Dropout → Dense (512) → Dense (7)

```

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) is a publicly available dataset with 35,887 grayscale 48x48 pixel facial images labeled with 7 emotion categories.

## Installation

#### 1. Clone the repository
        git clone https://github.com/prasanga73/Pytorch.git
        cd "Emotion Detector"

#### 3. Install required packages
        pip install -r requirements.txt

## Run the file
      conda activate (environment_name)
      python live.py 
      or
      streamlit run ImageClassifier.py