# 🩺 Diabetes Health Risk Predictor
### **An Ensemble Learning Approach to Chronic Disease Prediction**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](YOUR_STREAMLIT_URL_HERE)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](YOUR_KAGGLE_URL_HERE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](YOUR_GITHUB_URL_HERE)

---

## 📋 Project Overview
This project leverages Machine Learning to estimate an individual's risk of diabetes based on **21 health indicators** from the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)**. 

Unlike simple binary classifiers, this system uses a **Soft-Voting Ensemble** to provide a nuanced, percentage-based risk profile across three categories: **Healthy**, **Pre-Diabetic**, and **Diabetic**.

---

## 🛠️ Technical Stack

### **Core Programming & Environment**
* **Language:** `Python 3.10+`
* **Environment Management:** `pip` & `requirements.txt`
* **Version Control:** `Git` & `GitHub Desktop`

### **Data Engineering & Preprocessing**
* **Data Manipulation:** `Pandas`, `NumPy`
* **Imbalance Handling:** `Imbalanced-Learn` (SMOTE-Tomek Hybrid)
* **Pipeline Engineering:** `Scikit-Learn Pipeline`
* **Feature Selection:** `SelectKBest` (Chi-squared / Mutual Info)

### **Machine Learning Modeling**
* **Ensemble Framework:** `Scikit-Learn VotingClassifier` (Soft Voting)
* **Gradient Boosting:** `XGBoost` & `LightGBM`
* **Model Persistence:** `Joblib` (High-efficiency serialization)

### **Visualization & UI**
* **Web Framework:** `Streamlit`
* **EDA:** `Matplotlib`, `Seaborn`

---

## 🚀 Key Features & Methodology

### 1. Data Processing & Class Imbalance
The dataset contains over **250,000 records**, characterized by extreme class imbalance.
* **`SMOTE-Tomek` Hybrid:** Used to synthetically oversample the minority class while cleaning overlapping samples.
* **Feature Selection:** Identified the **12 most impactful features** (BMI, GenHlth, Age, etc.) to ensure model efficiency.

### 2. The Ensemble Brain
I implemented a `VotingClassifier` combining two high-performance Gradient Boosting machines:
* **`XGBoost`:** Handles non-linear relationships and tabular data effectively.
* **`LightGBM`:** Optimized for speed and large-scale data.
* **Soft Voting:** Outputs probability distributions for medical screening.

### 3. Model Performance
* **Multiclass AUC-ROC Score: `0.7584`**
* This score demonstrates robust class separation despite survey noise.

---

## 💻 How the App Works
1.  **User Input:** 12 health metrics collected via sidebar.
2.  **Pipeline Transformation:** Real-time scaling/imputation via `Scikit-Learn Pipeline`.
3.  **Inference:** Ensemble model generates a probability distribution.
4.  **Risk Advice:** Displays **Healthy %**, **Pre-Diabetic %**, and **Diabetic %**.

---

## 📁 Repository Structure
```text
├── app.py                      # Streamlit UI & Logic
├── requirements.txt            # Dependency list
├── diabetes_ensemble_model.pkl # Trained Soft-Voting Ensemble
├── selected_features.pkl       # Feature alignment file
└── README.md                   # Project documentation
