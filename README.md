# diabetes-risk-predictor
An ensemble ML model (XGBoost + LightGBM) to predict diabetes risk using CDC survey data
🩺 Diabetes Health Risk predictor 
An Ensemble Learning Approach to Chronic Disease Prediction
📋 Project Overview
This project leverages Machine Learning to estimate an individual's risk of diabetes based on 21 health indicators from the CDC's Behavioral Risk Factor Surveillance System (BRFSS). Unlike simple binary classifiers, this system uses a Soft-Voting Ensemble to provide a nuanced, percentage-based risk profile across three categories: Healthy, Pre-Diabetic, and Diabetic.

🛠️ Technical Stack
Language: Python 3.x

Modeling: XGBoost, LightGBM, Scikit-Learn

Data Handling: Pandas, NumPy, SMOTE (Imbalanced-Learn)

Deployment: Streamlit Cloud, GitHub Desktop

🚀 Key Features & Methodology
1. Data Processing & Class Imbalance
The dataset contains over 250,000 records, characterized by extreme class imbalance (majority healthy).

SMOTE-Tomek Hybrid: Used to synthetically oversample the minority "Pre-Diabetic" class while cleaning overlapping samples to improve boundary definition.

Feature Selection: Identified the 12 most impactful features (BMI, GenHlth, Age, etc.) using SelectKBest and correlation analysis to ensure model efficiency and UI simplicity.

2. The Ensemble Brain
I implemented a VotingClassifier combining two high-performance Gradient Boosting machines:

XGBoost: Excellent at handling tabular data and non-linear relationships.

LightGBM: Optimized for speed and handling large-scale datasets efficiently.

Soft Voting: Instead of a simple "yes/no," the model outputs the probability for each class, which is vital for medical screening applications.

3. Model Performance
Multiclass AUC-ROC Score: 0.7584

This score demonstrates the model’s robust ability to distinguish between distinct health profiles despite the inherent noise in survey data.

💻 How the App Works (Step-by-Step)
User Input: The user provides 12 health metrics via a sidebar (Age, BMI, Blood Pressure status, etc.).

Pipeline Transformation: The input data is automatically scaled and imputed using a saved Scikit-Learn Pipeline to match the training environment exactly.

Inference: The ensemble model processes the data and generates a probability distribution.

Risk Advice: The app displays three metrics (Healthy %, Pre-Diabetic %, Diabetic %) and provides a simplified medical advisory based on those risks.

📁 Repository Structure
Plaintext

├── app.py                      # Streamlit UI & Logic
├── requirements.txt            # Dependency list (sklearn 1.5.0+, etc.)
├── diabetes_ensemble_model.pkl # Trained Soft-Voting Ensemble
├── selected_features.pkl       # List of features used for UI alignment
└── README.md                   # Project documentation
