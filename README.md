# 📊 Predicting Telecom Customer Churn to Improve Customer Retention

---

## 📌 Project Snapshot  
- **Project Type:** Machine Learning (Classification)  
- **Domain:** Telecom / Customer Analytics  
- **Goal:** Predict churn + identify retention drivers  
- **Best Model:** Logistic Regression (L1)  
- **Key Metric Focus:** Recall  

---

## 1. Business Problem / Motivation  

Customer churn is a critical challenge in the telecom industry, as losing existing customers directly reduces revenue and increases the cost of acquiring new ones. Identifying at-risk customers early allows businesses to take targeted, proactive retention actions. This project aims to build a predictive model that accurately identifies customers likely to churn while providing actionable insights to support data-driven retention strategies.

---

## 2. Project Overview  

Developed a Logistic Regression model to predict customer churn using behavioral and account-level data. The dataset was preprocessed through data cleaning, categorical encoding, and feature engineering to improve signal quality. To address class imbalance, SMOTE and class weighting were applied to optimize the trade-off between precision and recall.

Model interpretability techniques, including global feature importance and local explanations, were used to identify key drivers of churn such as contract type, monthly charges, and referral status.

---

## 3. Data  

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3/data)  
- **Type:** Structured tabular data  
- **Size:** 7,043 customer records  

### Key Features:
- Tenure  
- MonthlyCharges / TotalCharges  
- Contract type  
- Internet service type  
- Payment method  
- Customer support usage  

**Target Variable:** Churn  

---

## 4. Data Preprocessing  

### Data Cleaning & Feature Selection
- Removed leakage features (Churn Score, Churn Reason, CLTV, etc.)
- Dropped geographic noise (City, Zip Code, Latitude/Longitude)
- Removed redundant age-related variables

### Feature Engineering
- Urban_Rural segmentation  
- Tenure buckets  
- Revenue per month  
- Interaction features (Charge × Tenure, Charge per GB, etc.)  
- Engagement score  

### Missing Values
- “No Internet” / “No Offer” imputation based on business logic  

### Encoding & Reduction
- Binary and categorical encoding applied  
- Removed multicollinearity (Total Charges, Total Revenue)

---

## 5. Exploratory Data Analysis (EDA)  

### Key Insights
- Tenure is the strongest churn predictor (~50% early churn vs ~7.7% long-term)
- Month-to-month contracts have highest churn (~45.8%)
- Higher monthly charges increase churn risk
- Automated payment methods reduce churn
- Internet users show higher churn concentration
- Satisfaction Score identified as leakage and removed

---

## 6. Modeling Approach  

### Baseline Model: Logistic Regression
- L1 and L2 regularization tested  
- L1 achieved best recall (~0.86)

### Advanced Model: XGBoost
- Captures nonlinear relationships and interactions  
- Strong performance but less interpretable  

---

## 7. Model Training & Optimization  

### Class Imbalance Handling
- SMOTE tested but replaced with class weighting  
- Final weights: {0: 1, 1: 2}

### Hyperparameter Tuning
- Logistic Regression: GridSearchCV (Best C = 0.1)
- XGBoost: tuned depth (6), learning rate (0.03), estimators (200), subsampling (0.7)

### Validation
- Cross-validation used for generalization  
- Leakage prevention applied (removed post-outcome variables)

---

## 8. Results Summary  

| Model | Recall (Churn) | Precision (Churn) | Strengths | Limitations |
|------|----------------|-------------------|------------|-------------|
| Logistic Regression (L1) | ~0.81 | ~0.63 | Highly interpretable | Slightly lower precision |
| Random Forest | ~0.59 | ~0.75 | High precision | Low recall |
| XGBoost | ~0.83–0.87 | ~0.57–0.61 | Strong performance | Lower interpretability |

---

## 9. Final Model Selection  

Logistic Regression (L1 with class weighting) was selected due to:
- Strong recall performance  
- High interpretability  
- Clear explainability via SHAP  

---

## 10. Model Interpretation (SHAP)  

### Global Insights
<img width="796" height="940" alt="Monthly Charge" src="https://github.com/user-attachments/assets/bf218d07-74f6-4752-a2a4-5aeec65aed46" />

- Referrals → strongest retention signal  
- Monthly charges → strongest churn driver  
- Contract type → reduces churn significantly  
- Tenure → strong retention factor  

### Local Insights
<img width="910" height="600" alt="2 159 - Number of Referrals" src="https://github.com/user-attachments/assets/439cdc2b-bd29-483b-89da-6ad0070a19c6" />

- High charges increase churn probability  
- Referrals and contracts reduce churn risk  
- Prediction reflects balance of behavioral signals  

---

## 11. Key Insights  

- Early lifecycle customers are highest churn risk  
- Contract duration is strongest retention lever  
- Pricing strongly drives churn  
- Engagement reduces churn risk  
- Behavioral features outperform demographics  

---

## 12. Business Impact & Recommendations  

### Key Drivers
- Contract type, monthly charges, tenure, referrals, age  

### Recommendations
- Improve onboarding (first 6 months critical)  
- Incentivize long-term contracts  
- Address pricing sensitivity  
- Strengthen engagement & referral programs  
- Target high-risk customers proactively  

---

## 13. Deployment Considerations  

- Inputs: tenure, monthly charges, contract type, referrals  
- Output: churn probability + risk classification  
- Use case: decision-support for retention teams  

Pilot strategy:
- Start with high-risk customers  
- Apply targeted retention campaigns  
- Scale after validation  

---

## 14. Expected Impact & Risks  

### Expected Impact
- Improved customer retention  
- Reduced revenue loss  
- More efficient targeting  

### Risks
- False negatives (missed churners)  
- False positives (extra cost)  
- Model drift over time  

### Mitigation
- Use as decision-support tool  
- Continuous monitoring  
- Periodic retraining  

---

## 15. Conclusion  

This project built an end-to-end churn prediction pipeline using machine learning. Logistic Regression was selected as the final model due to its balance of performance and interpretability. The model provides actionable insights into churn drivers and supports data-driven retention strategies.

---

## 16. Future Work  

- Deploy Streamlit dashboard  
- Real-time churn scoring  
- Advanced ensemble tuning  
- Time-series behavioral features  
- Model retraining pipeline  

---

## 17. How to Run  

```bash
pip install -r requirements.txt

jupyter notebook notebooks/01_data_understanding.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
jupyter notebook notebooks/03_baseline_modeling.ipynb
jupyter notebook notebooks/04_LogisticRegression_Modeling.ipynb
jupyter notebook notebooks/05_XGBoost_Modeling.ipynb
