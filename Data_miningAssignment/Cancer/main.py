# =========================================
# CANCER DATA CLASSIFICATION (BINARY - 2x2)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==============================
# 1. LOAD DATA (FIXED PATH)
# ==============================
print("Loading dataset...")

df = pd.read_excel(r"C:\Users\anish\OneDrive\Desktop\cancer\cancer patient data sets.xlsx")

print("Shape:", df.shape)
print(df.head())

# ==============================
# 2. DATA CLEANING
# ==============================

# Drop ID column
if "Patient Id" in df.columns:
    df = df.drop(columns=["Patient Id"])

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Convert target column safely
# Adjust labels based on your dataset values
df["Level"] = df["Level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 1
})

# Encode categorical features (SAFE WAY)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Handle missing values
print("\nMissing values:\n", df.isnull().sum())
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# ==============================
# 3. FEATURE SELECTION
# ==============================
X = df.drop(columns=["Level"])
y = df["Level"]

# ==============================
# 4. FEATURE SCALING (IMPORTANT)
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# ==============================
# 6. PERCEPTRON MODEL
# ==============================
perceptron = Perceptron(random_state=0)
perceptron.fit(X_train, y_train)

y_pred1 = perceptron.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)

print("\n=== PERCEPTRON RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Confusion Matrix:\n", cm1)
print(classification_report(y_test, y_pred1))

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.title("Perceptron Confusion Matrix (2x2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 7. LOGISTIC REGRESSION MODEL
# ==============================
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

y_pred2 = log_model.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Confusion Matrix:\n", cm2)
print(classification_report(y_test, y_pred2))

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Cancer", "Cancer"],
            yticklabels=["No Cancer", "Cancer"])
plt.title("Logistic Regression Confusion Matrix (2x2)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 8. CUSTOM PREDICTION
# ==============================
sample = X_test[0:1]
prediction = log_model.predict(sample)

print("\nSample Prediction:")
print("0 = No Cancer, 1 = Cancer")
print("Prediction:", prediction[0])