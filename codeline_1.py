import pandas as pd

df = pd.read_csv("netflix_customer_churn.csv")


print("Shape (rows, cols):", df.shape)
print("Columns:", list(df.columns))
print("First 5 rows:\n", df.head())
print("Churned value counts:\n", df["churned"].value_counts(dropna=False))

df = df.dropna()

feature_cols = ["age", "gender", "subscription_type", "watch_hours", "last_login_days", "region", "device", "monthly_fee", "payment_method", "number_of_profiles", "avg_watch_time_per_day", "favorite_genre"]

X = df[feature_cols]
y = df["churned"]

X = pd.get_dummies(X, drop_first=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

model = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print(scores)
print(scores.mean())
print(scores.std())

from sklearn.linear_model import LogisticRegression

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)

coefficients = pd.Series(final_model.coef_[0], index=X.columns).sort_values(ascending=False)
print(coefficients)


#NEW UNKNOWN DATAS ARE BELOW:- 

# Data number 1 for least risky customer

new_customer = pd.DataFrame([{
    "age": 32,
    "gender": "Male",
    "subscription_type": "Standard",
    "watch_hours": 120,
    "last_login_days": 2,
    "region": "Europe",
    "device": "TV",
    "monthly_fee": 12.99,
    "payment_method": "Debit Card",
    "number_of_profiles": 3,
    "avg_watch_time_per_day": 1.8,
    "favorite_genre": "Sci-Fi"
}])

new_customer_encoded = pd.get_dummies(new_customer)
new_customer_encoded = new_customer_encoded.reindex(columns=X.columns, fill_value=0)

churn_probability = final_model.predict_proba(new_customer_encoded)[0][1]
print("Churn probability:", churn_probability)


# Data number 2 for most risky customer

risky_customer = pd.DataFrame([{
    "age": 41,
    "gender": "Other",
    "subscription_type": "Premium",
    "watch_hours": 8,
    "last_login_days": 45,
    "region": "South America",
    "device": "Laptop",
    "monthly_fee": 19.99,
    "payment_method": "Crypto",
    "number_of_profiles": 1,
    "avg_watch_time_per_day": 0.05,
    "favorite_genre": "Drama"
}])

risky_customer_encoded = pd.get_dummies(risky_customer)
risky_customer_encoded = risky_customer_encoded.reindex(columns=X.columns, fill_value=0)

churn_probability_risky = final_model.predict_proba(risky_customer_encoded)[0][1]
print("Churn probability (risky customer):", churn_probability_risky)


# Data number 3 for medium risky customer


medium_risk_customer = pd.DataFrame([{
    "age": 38,
    "gender": "Male",
    "subscription_type": "Basic",
    "watch_hours": 45,
    "last_login_days": 12,
    "region": "Europe",
    "device": "Mobile",
    "monthly_fee": 9.99,
    "payment_method": "PayPal",
    "number_of_profiles": 2,
    "avg_watch_time_per_day": 0.7,
    "favorite_genre": "Comedy"
}])

medium_risk_encoded = pd.get_dummies(medium_risk_customer)
medium_risk_encoded = medium_risk_encoded.reindex(columns=X.columns, fill_value=0)

churn_probability_medium = final_model.predict_proba(medium_risk_encoded)[0][1]
print("Churn probability (medium risk):", churn_probability_medium)



# Data number 4 for medium risky customer


medium_risk_customer = pd.DataFrame([{
    "age": 38,
    "gender": "Male",
    "subscription_type": "Basic",
    "watch_hours": 20,
    "last_login_days": 18,
    "region": "Europe",
    "device": "Mobile",
    "monthly_fee": 9.99,
    "payment_method": "PayPal",
    "number_of_profiles": 1,
    "avg_watch_time_per_day": 0.25,
    "favorite_genre": "Comedy"
}])

medium_risk_encoded = pd.get_dummies(medium_risk_customer)
medium_risk_encoded = medium_risk_encoded.reindex(columns=X.columns, fill_value=0)

churn_probability_medium = final_model.predict_proba(medium_risk_encoded)[0][1]
print("Churn probability (medium risk):", churn_probability_medium)

# SIGMOID GRAPH
import numpy as np
import matplotlib.pyplot as plt

z = final_model.decision_function(X)
p = final_model.predict_proba(X)[:, 1]

idx = np.argsort(z)
z_sorted = z[idx]
p_sorted = p[idx]

plt.figure()
plt.plot(z_sorted, p_sorted, ".", markersize=2)
plt.xlabel("Model score (decision function)")
plt.ylabel("Predicted churn probability")
plt.title("Logistic Regression: Probability Mapping (from my data)")
plt.savefig("SIGMOID.png", dpi=300, bbox_inches="tight")

plt.show()

# ROC GRAPH

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_prob = final_model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Churn Prediction Model")
plt.legend()
plt.savefig("ROC.png", dpi=300, bbox_inches="tight")

plt.show()

