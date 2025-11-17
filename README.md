# AIML project Group-7_C1_Breast-cancer-Detection

# 🔬 Multimodal Breast Cancer Diagnostic Hub

Contributors: UCE2023605, UCE023623, UCE2023625

An integrated, interactive web application built with Streamlit for multimodal breast cancer analysis. This tool provides two distinct, parallel pipelines for diagnostic prediction: one for mammographic image analysis using Deep Learning, and one for tabular clinical data using classical Machine Learning.

The primary goal of this project is to create a holistic, trustworthy, and educational diagnostic aid. It mirrors a radiologist's real-world workflow by analyzing both images and data, while using **Explainable AI (XAI)** in *both* modules to build trust and validate model reasoning.

---

## ✨ Features

This application is split into two main modes:

### 1. 📷 Image Prediction (CNN)
* **Deep Learning Model:** Utilizes a **ResNet-50** Convolutional Neural Network (CNN), pre-trained on ImageNet and fine-tuned on the MIAS mammography dataset.
* **Instant Prediction:** Allows users to upload a mammogram (PNG/JPG) and receive an immediate "Benign" or "Malignant" classification.
* **Explainable AI (XAI):** Generates a **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmap for every prediction. This heatmap visually highlights *which parts* of the image the CNN focused on to make its decision, helping to identify if the model is "cheating" by looking at artifacts.
* **Performance Report:** Displays the model's full training history (Accuracy/Loss curves) and its final classification report.

### 2. 📊 CSV Data Analysis (Classical ML)
An end-to-end dashboard for analyzing tabular clinical data (e.g., the UCI Wisconsin Breast Cancer dataset).

* **Automated EDA:** On CSV upload, automatically generates:
    * A **Feature Correlation Heatmap** to show multicollinearity.
    * A **Hierarchical Clustering Dendrogram** to find groups of related features.
* **Model Training:** Trains a suite of 5 classical ML models (Random Forest, Decision Tree, SVM, KNN, Logistic Regression) using optimized, pre-discovered hyperparameters for a fast, "production-ready" demonstration.
* **XAI Dashboard:** A "Feature Importance" tab shows the top 10 most predictive features for each model.
* **Simulation:** A "Simulate New Patient" feature that predicts on a random sample from the dataset.
* **Architectural Comparison:** A dedicated tab that visually compares the internal logic of:
    * **Decision Tree** (plotting the full flowchart) vs. **Random Forest** (plotting a single estimator).
    * **CNN (Adam optimizer)** on *scaled* data vs. **CNN (SGD optimizer)** on *unscaled* data, complete with dynamic training graphs to demonstrate *why* data scaling is critical for neural networks.

---

## 💻 Technologies Used

* **Backend:** Python
* **Web Framework:** Streamlit
* **Deep Learning:** TensorFlow, Keras (for ResNet-50 & 1D CNNs)
* **Machine Learning:** Scikit-learn (for RF, DT, SVM, KNN, LR, PCA, K-Means)
* **Explainability:** `tf-keras-vis` (for Grad-CAM), `scikit-learn` (for Feature Importance)
* **Data Manipulation:** Pandas, NumPy
* **Plotting:** Matplotlib, Seaborn

---

## 🚀 Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is included.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create this file by running `pip freeze > requirements.txt` in your project environment)*

---

## 🏃‍♂️ How to Use

1.  **Run the Streamlit app:**
    Once all packages are installed, run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```

2.  **Select a Mode:**
    In the web app, choose either "Image Prediction (CNN)" or "CSV Data Analysis (Classical ML)".

3.  **Upload Your Data:**
    * **For Image Mode:** Upload a PNG or JPG mammogram.
    * **For CSV Mode:** Upload a compatible CSV file (like the included `data 3.csv`).

4.  **Explore the Tabs:**
    Navigate through the different tabs to see the EDA, train models, and view the architectural comparisons.

---
