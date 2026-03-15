# Machine-Learning-2025-26

# Motion Detection using Ultrasonic Sensor and Machine Learning

This project presents a motion detection system based on **ultrasonic echo signal analysis combined with machine learning techniques**. The objective of the system is to automatically distinguish between two environmental conditions:

- **No Movement (Static environment)**
- **Movement Detected (Dynamic environment)**

The system uses an **SRF02 FIUS Ultrasonic Sensor** connected to the **Red Pitaya STEMLab measurement platform** to collect ultrasonic echo signals. These signals are recorded and analyzed using signal processing techniques. Specifically, the recorded signals are transformed from the **time domain into the frequency domain using the Fast Fourier Transform (FFT)**. From the frequency spectrum, relevant statistical and spectral features are extracted.

These features are then used as inputs to several supervised machine learning algorithms to classify the environment as either **movement or no movement**. The implemented classifiers include **Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP)**.

The goal of this project is to demonstrate that **ultrasonic sensing combined with machine learning can serve as a reliable, low-cost, and privacy-preserving alternative to camera-based motion detection systems**.

---

# Project Overview

Motion detection is widely used in modern technological systems. Applications include:

- Smart home automation
- Security and surveillance systems
- Industrial monitoring
- Presence detection systems
- Human activity recognition

Most existing motion detection systems rely heavily on **camera-based technologies**. While cameras provide rich visual information, they introduce several limitations:

- **Privacy concerns** when monitoring people
- **High computational requirements** for image processing
- **Sensitivity to lighting conditions**
- **High hardware and processing cost**

To address these limitations, this project explores the use of **ultrasonic sensors** as an alternative sensing technology.

Ultrasonic sensors operate by transmitting **high-frequency sound waves** and receiving the reflected echoes from nearby objects. When an object moves in the sensor's detection area, the characteristics of the reflected signal change. These changes can be analyzed to determine whether motion has occurred.

However, ultrasonic signals can be affected by environmental noise, reflections, and signal distortions. Therefore, signal processing techniques and machine learning models are required to reliably detect motion.

This project focuses on:

1. Collecting ultrasonic echo data in both static and dynamic conditions.
2. Processing the signals using Fast Fourier Transform (FFT).
3. Extracting meaningful features from the signals.
4. Training machine learning models to classify movement conditions.

---

# Hardware Setup

The experimental hardware setup consists of three main components:

### SRF02 Ultrasonic Sensor

The **SRF02 ultrasonic sensor** is responsible for generating ultrasonic pulses and receiving reflected echo signals. Unlike traditional ultrasonic sensors that use separate transmitters and receivers, the SRF02 uses a **single transducer** for both transmitting and receiving signals.

Key characteristics of the sensor:

- Operates with a **5V power supply**
- Minimum sensing distance around **20 cm**
- Uses ultrasonic sound waves for object detection
- Suitable for distance measurement and motion detection

### Red Pitaya STEMLab Board

The **Red Pitaya STEMLab platform** acts as the signal acquisition and processing interface. It is based on a **Xilinx System-on-Chip architecture** and provides powerful measurement and signal processing capabilities.

Main functions of the Red Pitaya in this project:

- Capturing ultrasonic echo signals
- Converting analog signals to digital format
- Transmitting collected data to a computer
- Supporting real-time signal visualization

The board also includes:

- Analog RF input and output channels
- Ethernet connectivity
- USB interfaces
- Micro-SD card storage

### Laptop / Computer

A laptop or workstation is used to:

- Control the Red Pitaya system
- Record ultrasonic signal data
- Store measurement datasets
- Perform signal processing and machine learning analysis using Python

---

# Signal Processing

The ultrasonic sensor generates **time-domain signals** that represent the reflected echo waveform.

Each measurement contains thousands of signal samples, which makes direct analysis computationally expensive and inefficient for machine learning models.

To address this problem, the signals are transformed using the **Fast Fourier Transform (FFT)**.

### Fast Fourier Transform (FFT)

The Fast Fourier Transform converts a signal from the **time domain** into the **frequency domain**.

This transformation allows us to analyze the frequency components of the signal and identify patterns associated with movement.

Benefits of using FFT:

- Reduces computational complexity
- Reveals dominant frequency components
- Simplifies feature extraction
- Improves machine learning performance

### Extracted Features

From the FFT spectrum, several statistical features are extracted:

| Feature | Description |
|------|------|
| fft_mean | Mean magnitude of the frequency spectrum |
| fft_std | Standard deviation of spectral values |
| fft_max | Maximum spectral magnitude |
| fft_energy | Total energy of the spectrum |
| fft_peak_idx | Index of the highest frequency peak |

These features represent important characteristics of the ultrasonic signal while significantly reducing the dimensionality of the dataset.

---

# Dataset

The dataset used in this project consists of ultrasonic signal measurements collected during controlled experiments.

Each measurement corresponds to one ultrasonic scan recorded from the sensor.

### Dataset Format

The recorded data is stored in **CSV files** containing time-series signal samples.

Each row in the dataset contains:

- Raw ultrasonic signal samples
- Sensor measurement information
- Class label indicating motion condition

### Class Labels

| Label | Description |
|------|------|
| 0 | No Movement (NO-MOV) |
| 1 | Movement Detected (MOV) |

### Dataset Size

The final dataset contains approximately **30,000 signal samples**, equally distributed between the two classes.

This balanced dataset improves the robustness and reliability of the machine learning models.

---

# Machine Learning Models

Four supervised learning algorithms were implemented and compared.

### Logistic Regression

Logistic Regression is a **linear classification algorithm** that predicts the probability of a sample belonging to a specific class.

It uses the **sigmoid function** to map input features into probability values between 0 and 1.

Although it is computationally efficient, logistic regression struggles to capture complex nonlinear patterns in ultrasonic signals.

---

### K-Nearest Neighbors (KNN)

KNN is a **non-parametric classification algorithm** that classifies a data point based on the labels of its nearest neighbors.

Key properties:

- Uses distance metrics (usually Euclidean distance)
- Does not require explicit model training
- Works well with nonlinear decision boundaries

In this project, KNN demonstrated **the highest classification accuracy** among all tested models.

---

### Support Vector Machine (SVM)

Support Vector Machines aim to find the **optimal separating hyperplane** that maximizes the margin between different classes.

When data cannot be separated linearly, kernel functions are used to transform the data into a higher-dimensional feature space.

In this implementation, the **Radial Basis Function (RBF) kernel** is used to capture nonlinear patterns in the FFT features.

---

### Multi-Layer Perceptron (MLP)

The Multi-Layer Perceptron is a **feedforward artificial neural network** composed of multiple layers:

- Input layer
- One or more hidden layers
- Output layer

MLP can learn complex nonlinear relationships between input features and output classes.

In this project, the MLP model contains hidden layers with **64 and 32 neurons**, enabling it to learn complex signal patterns.

---

# Model Evaluation

To evaluate the classification performance of the models, several evaluation metrics are used.

### Confusion Matrix

A confusion matrix summarizes the prediction results of a classification model.

It consists of four values:

| Metric | Meaning |
|------|------|
| True Positive (TP) | Movement correctly detected |
| True Negative (TN) | No movement correctly detected |
| False Positive (FP) | Movement predicted incorrectly |
| False Negative (FN) | Movement missed by the model |

### Evaluation Metrics

The following metrics are computed from the confusion matrix:

- **Accuracy** – overall classification correctness
- **Precision** – proportion of correctly predicted movement events
- **Recall** – ability to detect actual movement
- **F1-Score** – harmonic mean of precision and recall

These metrics provide a comprehensive understanding of model performance.

---

# Results

The classification performance of the evaluated models is summarized below.

| Model | Accuracy |
|------|------|
| KNN | **92.63%** |
| MLP | 92.21% |
| SVM | 88.91% |
| Logistic Regression | 81.65% |

### Key Observations

- **KNN achieved the best overall performance**
- **MLP performed similarly well due to its nonlinear learning capability**
- **SVM performed moderately well**
- **Logistic Regression showed lower performance due to its linear decision boundary**

These results demonstrate that **nonlinear machine learning models are more suitable for ultrasonic signal classification tasks**.

---


## Repository Structure

```
Machine-Learning-2025-26/
│
├── MOV_Code/
│   ├── mov_ml_knn_fft.py
│   ├── mov_ml_lr_fft.py
│   ├── mov_ml_mlp_fft.py
│   └── mov_ml_svm_fft.py
│
├── Images/
│   └── Image_FFT/
│       └── header_for_None/
│           ├── Accuracy_Knn_Fft.png
│           ├── Accuracy_Lr_Fft.png
│           ├── Accuracy_Mlp_Fft.png
│           ├── Accuracy_Svm_Fft.png
│           ├── confusion_matrix_Knn_Fft.png
│           ├── confusion_matrix_Lr_Fft.png
│           ├── confusion_matrix_Mlp_Fft.png
│           └── confusion_matrix_Svm_Fft.png
│
└── README.md
```

---

# Technologies Used

The following technologies were used to implement the project:

- Python
- NumPy
- Pandas
- Scikit-learn
- Fast Fourier Transform (FFT)
- Red Pitaya STEMLab platform
- Ultrasonic sensing technology

---

# Limitations

Although the system performs well in controlled environments, several factors may influence its performance in real-world scenarios.

Key challenges include:

- Environmental noise and signal interference
- Reflections from multiple surfaces
- Temperature and humidity variations
- Object distance and orientation relative to the sensor

Future work will focus on improving system robustness under these conditions.

---

# Future Work

Possible improvements to the system include:

- Implementing **real-time motion detection**
- Deploying the model on **embedded edge devices**
- Expanding the dataset with additional environmental conditions
- Combining ultrasonic sensing with other sensors
- Optimizing model performance for real-time applications

---

# Authors

Kumar Satyam  
Pradeep Maheshnarayan Tiwari  
Rajan Vijakumar Singh

Frankfurt University of Applied Sciences  
Machine Learning – Winter Semester 2025/2026

Supervisor: **Prof. Dr. Andreas Pech**

---

# References

1. Red Pitaya STEMLab Documentation  
2. SRF02 Ultrasonic Sensor Technical Documentation  
3. Bishop, C. M. *Pattern Recognition and Machine Learning*  
4. Hastie, Tibshirani, Friedman – *The Elements of Statistical Learning*
5. R. B. Saleem Basha, S. Veerapur, “Parameter setting and reliability test of a sensor system for infant carrier car seat sensing in a car using a dashboard sensor.” Student Project Report, Frankfurt University of Applied Sciences, Frankfurt, Germany.
