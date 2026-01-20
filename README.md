# Netflix Customer Churn Prediction

This repository contains a machine learning project for predicting customer churn on a Netflix-like OTT platform using Logistic Regression.

---

## Dataset Description

The dataset used in this project is a **synthetic customer churn dataset** designed to simulate realistic OTT platform user behavior.

- **Total records:** ~5,000 customers
- **Target variable:** `churned` (0 = retained, 1 = churned)
- **Format:** CSV

📄 Dataset file:  
`netflix_customer_churn.csv`

---

## Dataset Structure (Feature Overview)

| Feature Name | Description |
|-------------|------------|
| customer_id | Unique customer identifier |
| age | Age of the customer |
| gender | Gender category |
| subscription_type | Subscription plan (Basic / Standard / Premium) |
| watch_hours | Total hours watched |
| last_login_days | Days since last login |
| region | Geographic region |
| device | Primary device used |
| monthly_fee | Subscription cost |
| payment_method | Payment method used |
| number_of_profiles | Number of profiles on account |
| avg_watch_time_per_day | Average daily watch time |
| favorite_genre | Most watched genre |
| churned | Target variable (1 = churned, 0 = retained) |

---

## Methodology

- Data preprocessing using one-hot encoding
- Logistic Regression model
- 10-fold Stratified Cross-Validation
- Evaluation using ROC-AUC metric

---

## Code

Main implementation file:
- `codeline_1.py`

---

## Results

- Mean ROC-AUC (10-fold CV): **~0.97**
- Engagement and inactivity are the strongest churn drivers
