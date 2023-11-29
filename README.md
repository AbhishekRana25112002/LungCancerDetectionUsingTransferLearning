# Lung Cancer Prediction using Transfer Learning

## Overview

This project focuses on predicting lung cancer based on histopathological images using deep learning with transfer learning. The dataset used for this project includes lung cancer images, categorized into different classes. The InceptionV3 model is utilized for feature extraction, and a custom neural network is added for classification.

## Dataset

The histopathological images dataset used in this project can be found on [Kaggle](https://www.kaggle.com/andrewmvd/lung-and-colon-cancer-histopathological-images). It includes images related to lung cancer.

## Tools and Technologies

- **InceptionV3**: A pre-trained convolutional neural network (CNN) used for feature extraction.
- **TensorFlow and Keras**: Deep learning libraries for model development and training.
- **NumPy, Pandas, and Matplotlib**: Python libraries for data manipulation and visualization.
- **OpenCV**: Computer vision library used for image processing.
- **Google Colab**: The notebook is created and executed in a Google Colab environment.

## Data Preparation

The dataset is loaded, and images are resized and preprocessed for training. Data is split into training and validation sets.

## Model Architecture

Transfer learning is employed using the InceptionV3 model with added dense layers for classification. The model is compiled using the Adam optimizer and categorical crossentropy loss.

## Model Training

The model is trained on the prepared dataset, and a custom callback is used to stop training when validation accuracy reaches 90%. Training metrics are visualized to monitor the model's performance.

## Model Evaluation

The trained model is evaluated on the validation dataset, and metrics such as confusion matrix and classification report are provided. The model's ability to predict lung cancer classes is assessed.

## Usage

1. Open the provided Jupyter notebook `LungCancerPredTL.ipynb` in Google Colab.
2. Execute the notebook cells sequentially to load the data, prepare it, define and train the model, and evaluate the results.

## Model File

The trained lung cancer prediction model is saved as `lung_cancer_detection_model.h5` for future use.

## Results

Training and validation loss and accuracy curves are visualized. Confusion matrix and classification report provide insights into the model's performance.


