# Malaria Detection Project

## Project Overview
This project aims to automate the detection of malaria in blood smear images using deep learning techniques. The project focuses on **loading the dataset, preprocessing images, and visualizing results**. It forms part of a larger pipeline for malaria diagnosis.

---

## Your Role in the Project
As assigned, the responsibilities include:
- **Dataset Collection & Loading:** Downloading and organizing the Malaria Cell Images Dataset.  
- **Data Preprocessing:** Resizing, normalizing, and preparing images for training.  
- **Data Visualization:** Plotting training and validation accuracy/loss, confusion matrices, and ROC-AUC for model evaluation.  

> Note: Model saving, building an application, or deployment is **not included** in this role.

---

## Dataset
- **Name:** Malaria Cell Images Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)  
- **Total Images:** 27,558  
- **Classes:**  
  - Parasitized (0) – infected red blood cells  
  - Uninfected (1) – healthy red blood cells  
- **Image Properties:**  
  - Thin blood smear microscopy images  
  - RGB color images  
  - Dimensions: Average 141×143 pixels (varies from 79×82 to 247×226)  
  - Staining method: Giemsa staining  

> **Note:** For this project, the full dataset is used locally, but **dataset files are not uploaded to GitHub** due to size constraints.

---

## Libraries Used
The following Python libraries are required:

- `numpy` – numerical computation and array manipulation  
- `opencv-python` – image loading, resizing, and preprocessing  
- `matplotlib` – plotting graphs and visualization  
- `seaborn` – advanced visualization (heatmaps for confusion matrix)  
- `tensorflow` – deep learning framework for model creation and training  
- `scikit-learn` – train-test split, evaluation metrics  

These libraries are also listed in `requirements.txt` for easy installation.

---

## Files in This Repository
- **train.py** – Main Python script that performs:
  - Loading and preprocessing dataset  
  - Data visualization (accuracy/loss plots, confusion matrix, ROC-AUC)  
  - Training models (Custom CNN, MobileNetV2, EfficientNetB0) for visualization purposes  

> The code does **not save model files** or build a user interface, as it is outside the assigned role.

---

## How to Run
1. Ensure Python 3.x is installed on your system.  
2. Install required libraries (preferably in a virtual environment):

```bash
pip install -r requirements.txt