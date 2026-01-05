# Malware Detection Using Convolutional Neural Networks

**Lecturer:** Dr. Arnaud Fadja  
**Student:** TENE TCHIO FRANCK DE PADOUE  
**Student ID:** UBa24EP011  
**Course:** Msc in Cybersecurity and Cryptology

---

## 1. Project Overview
This project aims to develop a robust deep learning pipeline for image-based malware detection. By converting malware binaries into spatial pixel patterns, we utilize **Convolutional Neural Networks (CNNs)** to learn texture features that distinguish malicious software from benign applications.

The project demonstrates:
* **Data Augmentation:** Techniques to improve model generalization.
* **Hyperparameter Tuning:** Systematic optimization of learning rates and architectures.
* **Model Comparison:** Performance benchmarking between a **Custom CNN** and a transfer learning approach using **VGG16**.

## 2. Dataset Description
The "Malware Benign Image Sample Dataset" contains **22,056 samples** represented as grayscale images.
* **Format:** $128 \times 128$ pixel grids.
* **Classes (8):** Adware, Backdoor, Benign, Downloader, Spyware, Trojan, Virus, and Worms.
* **Distribution:** * Training: 14,331 images
    * Validation: 3,304 images
    * Test: 4,418 images

## 3. Methodology & Architectures
The pipeline follows these core steps:
1. **Preprocessing:** Image scaling and normalization.
2. **Custom CNN Architecture:** * Multiple `Conv2D` layers with `ReLU` activation.
    * `MaxPooling2D` for dimensionality reduction.
    * `Dropout` (0.5) to prevent overfitting.
    * Final `Dense` layer with `Softmax` for multi-class classification.
3. **Transfer Learning (VGG16):** Utilizing a pre-trained VGG16 backbone on ImageNet, with custom top layers for the 8 malware classes.
4. **Optimization:** Use of `Adam` optimizer and `ReduceLROnPlateau` scheduler.

## 4. Key Results
Based on the experiments conducted in the notebook:
* **Best Model:** VGG16 (Transfer Learning) provided the highest test accuracy.
* **Performance:** The models successfully identified complex malware patterns that are otherwise difficult to detect via traditional signature-based methods.
* **Inference:** Data augmentation proved essential in stabilizing the training loss.

## 5. How to Run
### Prerequisites
* Python 3.10+
* TensorFlow 2.x
* Matplotlib / Seaborn
* GPU access (Google Colab T4 recommended)

### Setup
1. Mount your Google Drive or upload the dataset to your local environment.
2. Run the notebook `TENE_FRANCK_UBa24EP011.ipynb`.
3. The final comparison table and confusion matrices will be generated at the end of the execution.

---
© 2025 - University of Bamenda - Centre for Cybersecurity and Mathematical Cryptology
