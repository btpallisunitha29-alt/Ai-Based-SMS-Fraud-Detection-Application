# ============================================================
# SMS FRAUD DETECTION
# STEP 2: LINEAR SVM
#
# Input:
#   transformer_features.npy
#   cleaned_sms_dataset.csv
#
# Process:
#   DistilBERT Features → Linear SVM
#
# Output:
#   Safe / Spam (Fraud)
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

FEATURE_FILE = r"C:\sms_project\data\transformer_features.npy"

DATA_FILE = r"C:\sms_project\data\cleaned_sms_dataset.csv"

MODEL_FILE = r"C:\sms_project\data\linear_svm_model.pkl"


# ============================================================
# 2. START
# ============================================================

print("=" * 60)
print("SMS FRAUD DETECTION")
print("DISTILBERT FEATURES → LINEAR SVM")
print("=" * 60)


# ============================================================
# 3. CHECK FILES
# ============================================================

if not os.path.exists(FEATURE_FILE):

    print("\nERROR: Transformer output not found!")
    print(FEATURE_FILE)
    print("\nRun transform.py first.")
    exit()


if not os.path.exists(DATA_FILE):

    print("\nERROR: Cleaned dataset not found!")
    print(DATA_FILE)
    exit()


print("\nTransformer output found.")
print("Cleaned dataset found.")


# ============================================================
# 4. LOAD TRANSFORMER OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRANSFORMER OUTPUT")
print("=" * 60)

X = np.load(FEATURE_FILE)

print("\nTransformer features loaded.")

print("Feature shape:", X.shape)

print("Number of SMS:", X.shape[0])

print("Number of features:", X.shape[1])


# ============================================================
# 5. LOAD LABELS
# ============================================================

print("\n" + "=" * 60)
print("LOADING LABELS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

print("Columns:", df.columns.tolist())


# ============================================================
# 6. FIND LABEL COLUMN
# ============================================================

label_column = None

for column in [
    "Label",
    "label",
    "Category",
    "category",
    "Class",
    "class"
]:

    if column in df.columns:

        label_column = column
        break


if label_column is None:

    print("\nERROR: Label column not found.")

    print("Available columns:")
    print(df.columns.tolist())

    exit()


print("\nLabel column:", label_column)


# ============================================================
# 7. CONVERT LABELS
# ============================================================

labels = df[label_column].astype(str).str.strip().str.lower()

y = labels.map({

    "ham": 0,
    "safe": 0,
    "0": 0,

    "spam": 1,
    "fraud": 1,
    "1": 1

})


# ============================================================
# 8. REMOVE UNKNOWN LABELS
# ============================================================

valid = y.notna()

y = y[valid].astype(int).to_numpy()


# ============================================================
# 9. CHECK FEATURE/LABEL COUNT
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DATA")
print("=" * 60)

print("\nTransformer features:", len(X))

print("Labels:", len(y))


if len(X) != len(y):

    print("\nERROR: Feature count and label count do not match.")

    print("Features:", len(X))
    print("Labels:", len(y))

    exit()


print("\nFeature count and label count match.")


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# 11. CREATE LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("LINEAR SVM")
print("=" * 60)

svm = LinearSVC(

    C=1.0,

    random_state=42,

    max_iter=10000
)


print("\nLinear SVM created.")


# ============================================================
# 12. SEND TRANSFORMER OUTPUT TO SVM
# ============================================================

print("\nSending DistilBERT numerical features to Linear SVM...")

svm.fit(

    X_train,

    y_train
)


print("\nLinear SVM training completed.")


# ============================================================
# 13. PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

y_pred = svm.predict(X_test)


print("\nPredictions completed.")


# ============================================================
# 14. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall   : {recall * 100:.2f}%")

print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(

    y_test,

    y_pred
)

print(cm)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Safe/Ham",
            "Spam/Fraud"
        ],

        zero_division=0
    )
)


# ============================================================
# 17. SAVE LINEAR SVM MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING LINEAR SVM MODEL")
print("=" * 60)

joblib.dump(

    svm,

    MODEL_FILE
)


print("\nModel saved successfully:")

print(MODEL_FILE)


# ============================================================
# 18. FINAL PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)

print("""
SMS TEXT
   ↓
DistilBERT
   ↓
Numerical Features
   ↓
transformer_features.npy
   ↓
Linear SVM
   ↓
Safe / Spam (Fraud)
""")

print("Accuracy :", f"{accuracy * 100:.2f}%")
print("Precision:", f"{precision * 100:.2f}%")
print("Recall   :", f"{recall * 100:.2f}%")
print("F1 Score :", f"{f1 * 100:.2f}%")

print("\nLinear SVM model:")
print(MODEL_FILE)

print("=" * 60)