# CIFAR-10 Image Classification using ResNet50 (Transfer Learning)

A deep learning project that applies transfer learning with a pretrained **ResNet50** model to classify images from the **CIFAR-10** dataset. This work replicates and improves upon results from an existing research paper on CIFAR-10 classification.

---

## 🔗 Links

- 📓 Google Colab Notebook https://colab.research.google.com/drive/1-a_W0GP0FG-r690rnJ438YnFfrpVeH8R?usp=sharing)
- 📄 Research Paper Reference:  https://drive.google.com/drive/folders/1RYYN1y-0Mv3-W3YFJDLLTh2jxNCujhiA?usp=drive_link
- 📦 CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

---

## 📋 Project Overview

| Property | Details |
|---|---|
| **Dataset** | CIFAR-10 (60,000 images, 10 classes) |
| **Pretrained Model** | ResNet50 (ImageNet weights) |
| **Input Size** | 64×64×3 |
| **Framework** | TensorFlow / Keras |
| **Environment** | Google Colab (T4 GPU) |

---

## 📁 Dataset

- **Name:** CIFAR-10
- **Source:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Description:** 60,000 color images (32×32) across 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split Used:**

| Split | Samples |
|---|---|
| Training | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |

---

## ⚙️ Preprocessing

- Resized images from **32×32 → 64×64** (memory-safe for free Colab T4 GPU)
- Applied **ResNet-specific preprocessing** via `keras.applications.resnet50.preprocess_input()` (ImageNet mean subtraction)
- One-hot encoded labels for `categorical_crossentropy` loss
- Stratified train/validation split to preserve class balance

---

## 🏗️ Model Architecture

```
ResNet50 (pretrained, ImageNet)
    └── GlobalAveragePooling2D
    └── BatchNormalization
    └── Dense(256, relu)
    └── Dropout(0.4)
    └── Dense(128, relu)
    └── Dropout(0.3)
    └── Dense(10, softmax)
```

- First 155 layers **frozen** (preserves ImageNet features)
- Last 20 layers **fine-tuned** on CIFAR-10

---

## 🔧 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 20 |
| Early Stopping Patience | 5 |
| LR Reduce Patience | 3 |
| LR Reduce Factor | 0.5 |

---

## 🛠️ How to Run

1. Open the (https://colab.research.google.com/drive/1-a_W0GP0FG-r690rnJ438YnFfrpVeH8R?usp=sharing)
2. Mount your Google Drive when prompted
3. Run all cells in order — dataset loads automatically via `keras.datasets.cifar10`
4. Results and plots are generated at the end of Task 3

---

## 📦 Requirements

```
tensorflow >= 2.x
numpy
matplotlib
scikit-learn
seaborn
pandas
Pillow
```

All pre-installed in Google Colab — no manual installation needed.
