🩺 Diabetes Prediction and Management System

A Clinical Machine Learning–Based Risk Assessment Framework

📌 Overview

Diabetes mellitus is one of the fastest-growing chronic diseases in India, affecting millions of people and significantly reducing quality of life due to long-term complications. Early detection and risk-based intervention are critical to reducing disease burden.

This project implements an end-to-end machine learning pipeline for diabetes risk prediction and clinical decision support, using Logistic Regression for its interpretability, stability, and suitability in healthcare applications.

The system performs:

Synthetic clinical data generation

Exploratory data analysis (EDA)

Model training and evaluation

Explainable risk prediction

Clinical decision support simulation

🎯 Objectives

Predict diabetes risk using clinical and lifestyle parameters

Provide probability-based risk stratification

Support early screening and intervention decisions

Ensure interpretability and clinical relevance

Demonstrate an end-to-end ML healthcare pipeline

🧠 Why Logistic Regression?

Logistic Regression was chosen because:

Produces interpretable coefficients (clinically explainable)

Outputs well-calibrated probabilities

Works effectively on structured medical data

Preferred in healthcare for transparency and governance

Suitable for threshold-based clinical decision systems

This project prioritizes clinical trust over black-box accuracy.

🏗️ System Architecture
Clinical Data
(Age, BMI, Glucose, Lipids, Lifestyle)
        │
        ▼
Data Preprocessing
• Feature selection
• Standard scaling
        │
        ▼
Exploratory Data Analysis
• Distribution
• Correlation
• Visualization
        │
        ▼
Train–Test Split (80:20)
        │
        ▼
Logistic Regression Model
        │
        ▼
Model Evaluation
• Accuracy
• ROC–AUC
• Precision–Recall
        │
        ▼
Clinical Decision Support
• Low Risk
• Moderate Risk
• High Risk

📊 Dataset Description

Due to privacy constraints in healthcare data, a synthetic clinical dataset was generated using statistically realistic distributions.

Attribute	Value
Patients	1500
Features	12 clinical + 1 risk score
Target Variable	Diabetes Diagnosis (Binary)
Diabetes Prevalence	~25%
Data Type	Numerical + Binary
Clinical Features Used

Age

BMI

Glucose

Blood Pressure

Insulin

Skin Thickness

HDL Cholesterol

LDL Cholesterol

Triglycerides

Family History

Physical Activity

Diet Quality

🔬 Methodology

Data Generation
Synthetic patient profiles generated using real-world clinical ranges.

Exploratory Data Analysis (EDA)

Class distribution

Feature correlations

Risk score analysis

Preprocessing

Feature scaling using StandardScaler

Stratified train–test split

Model Training

Logistic Regression (LBFGS solver)

5-fold cross-validation

Evaluation Metrics

Accuracy

ROC–AUC

Precision, Recall, F1-Score

Clinical Decision Support
Probability-based risk categorization with actionable recommendations.

📈 Results
Model Performance
Metric	Value
Test Accuracy	0.893
ROC–AUC	0.957
CV Accuracy	~0.89
Diabetes Prevalence	25%
Classification Report
Class	Precision	Recall	F1-Score
Non-Diabetic	0.93	0.93	0.93
Diabetic	0.79	0.77	0.78
🔎 Feature Importance (Top Predictors)
Feature	Coefficient	Interpretation
Triglycerides	+2.31	Strong metabolic risk
Glucose	+2.01	Primary diabetes indicator
Insulin	+1.37	Insulin resistance
LDL Cholesterol	+0.94	Cardiovascular risk
HDL Cholesterol	−0.63	Protective factor
🧑‍⚕️ Clinical Decision Support Logic
If Probability < 0.30
   → LOW RISK
   → Annual Screening

If 0.30 ≤ Probability < 0.70
   → MODERATE RISK
   → Lifestyle Modification
   → 6-Month Follow-Up

If Probability ≥ 0.70
   → HIGH RISK
   → Immediate Clinical Evaluation

Sample Case Results
Case	Probability	Risk Level
Case 1	0.000	Low
Case 2	0.999	High
Case 3	0.309	Moderate
💡 Key Contributions

End-to-end ML healthcare pipeline

Clinically interpretable model

Probability-based risk stratification

Decision support rather than mere prediction

Designed for early screening use cases

⚠️ Limitations

Dataset is synthetic (no real patient data)

External clinical validation required

Not a diagnostic tool

Should complement medical testing, not replace it

🚀 Future Enhancements

Validation on real hospital datasets

Comparison with Random Forest / XGBoost

Model calibration and fairness analysis

Web or mobile-based clinical interface

Integration with EHR systems

Longitudinal risk prediction

🛠️ Tech Stack

Python

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

▶️ How to Run
# Clone repository
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
