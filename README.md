# brainTumor
A deep learning project for automated brain tumor detection from MRI images. It classifies tumors into glioma, meningioma, pituitary, or non-tumor categories and provides detailed information on each type, including description, treatment, and precautions. This tool aims to support early diagnosis and medical decision-making.

# Brain Tumor Detection Using AI

This project aims to build an advanced **Brain Tumor Classification System** using deep learning techniques to analyze **MRI scan images**. By leveraging two powerful convolutional neural network architectures, **ResNet50** and **DenseNet121**, along with custom CNN layers, the system classifies brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor. The project showcases the application of **Artificial Intelligence (AI)** in healthcare to help medical professionals detect tumors efficiently.


## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Testing and Evaluation](#testing-and-evaluation)
6. [Web Interface](#web-interface)
7. [Usage](#usage)


## Overview

This project provides an AI-based solution for detecting and classifying brain tumors from **MRI images** into four types:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

Using two state-of-the-art deep learning models, **ResNet50** and **DenseNet121**, the system has achieved impressive classification results:

- **ResNet50 + Custom CNN**: 96% accuracy
- **DenseNet121 + Custom CNN**: 99.69% accuracy

In addition to high accuracy, we have integrated multiple testing methods to evaluate and validate the models. A **web-based interface** was developed to allow users to upload MRI scans, detect tumors, and download detailed classification reports.

## Dataset

The dataset used in this project consists of **7,023 MRI images** of human brain scans. These images are categorized into four classes:

1. **Glioma**: A malignant tumor that originates in the glial cells of the brain.
2. **Meningioma**: A tumor that develops from the meninges, the protective layers surrounding the brain and spinal cord.
3. **No Tumor**: MRI images that show no evidence of a tumor.
4. **Pituitary Tumor**: A tumor that arises from the pituitary gland, located at the base of the brain.

### Dataset Details:
- **Number of Images**: 7,023 images
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Resolution**: The images vary in resolution but were resized to a consistent dimension (e.g., 224x224 pixels) for model input.
- **Image Format**: JPG/PNG
- **Annotations**: Each image is labeled with one of the four tumor categories.

### Data Preprocessing:
- **Resizing**: All images were resized to **224x224 pixels** to standardize the input size.
- **Normalization**: Pixel values were scaled to the range of **[0, 1]** to improve model training.
- **Augmentation**: To improve generalization and reduce overfitting, data augmentation techniques such as rotation, flipping, and zooming were applied to the training images.

## Model Architecture

We used two primary model architectures for classification:

### 1. **ResNet50 + Custom CNN**
- **ResNet50**: A deep residual network architecture that allows for very deep networks while solving the vanishing gradient problem using skip connections.
- **Custom CNN**: A series of convolutional layers added on top of ResNet50 to optimize feature extraction and enhance classification accuracy.
  
  **Accuracy**: **96%**

### 2. **DenseNet121 + Custom CNN**
- **DenseNet121**: A dense convolutional network that connects each layer to every other layer, improving feature reuse and reducing overfitting.
- **Custom CNN**: This component augments DenseNet121 by adding specialized convolutional layers for improved classification.

  **Accuracy**: **99.69%**

Both models were trained for **25 epochs**, optimizing the hyperparameters such as batch size, learning rate, and number of layers to achieve the best results.

### Model Performance Evaluation:
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam Optimizer with learning rate decay
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Training Procedure

The training process involved several key steps:

1. **Data Preparation**:
   - The images were divided into **training**, **validation**, and **test** sets.
   - The training set consisted of 80% of the dataset, while the validation and test sets each contained 10% of the data.

2. **Model Training**:
   - Both models were trained using **GPU** acceleration to speed up the training process.
   - The Adam optimizer was used with a learning rate of **0.0001**, and the batch size was set to **32**.
   - **Early stopping** was implemented to prevent overfitting, stopping the training process when validation accuracy stopped improving.

3. **Model Evaluation**:
   - After training, both models were evaluated on the test dataset.
   - Model performance was evaluated based on various metrics, including accuracy, precision, recall, and F1-score.

4. **Results**:
   - **ResNet50 + Custom CNN** achieved **96% accuracy** on the test dataset.
   - **DenseNet121 + Custom CNN** achieved **99.69% accuracy**, providing significantly better performance.

## Testing and Evaluation

We implemented multiple testing methods to validate model performance:

### 1. **Bulk Image Testing**:
- The models were tested on the entire dataset to evaluate the tumor detection capabilities on a large scale.
- Results were saved in a **CSV file**, including the tumor type and model confidence score for each image.

### 2. **Single Image Testing**:
- Users can input a single MRI image, and the model will predict the tumor type.
- The model provides the classification results, including the tumor type and confidence score.

### 3. **Voice-Assisted Detection**:
- Integrated a **text-to-speech (TTS)** engine that announces the tumor detection results for single-image testing.
- The TTS system also provides additional details about the detected tumor type, making the tool more interactive and accessible.

### 4. **Web-Based Interface**:
- A dynamic web application allows users to upload MRI scans, input patient details, and detect tumors.
- The web interface supports:
  - **Patient Details**: Name, age, sex, MRI scan, and clinical history.
  - **Detect Button**: Initiates the tumor detection process.
  - **Detailed Report**: After processing, a detailed tumor classification report is generated and available for download.

## Web Interface

The web interface was designed to be user-friendly and accessible. It includes the following features:
1. **Input Section**:
   - Users enter patient information (name, age, sex, MRI scan, clinical history).
2. **Detection Process**:
   - Press the **"Detect"** button to begin the tumor detection process.
3. **Report Generation**:
   - A detailed report is generated, including tumor type, classification results, and confidence score.
4. **Download Report**:
   - The user can download the report for medical use.
     
## Usage

1. Launch the web application by running `TESTING_4.py`.(It use streamlit for web page )
2. Enter patient details (name, age, MRI scan) and click **"Detect"** to process the image.
3. View and download the generated report containing tumor detection results.



### 1. Clone the Repository:
```bash
git clone https://github.com/RAJEEVRANJAN0001/brainTumor.git
cd brainTumor
