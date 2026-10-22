import os
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


# ============================================================
# SMS FRAUD DETECTION
# TRANSFORMER + LINEAR SVM TRAINING
# ============================================================

print("=" * 70)
print("        SMS FRAUD DETECTION - MODEL TRAINING")
print("=" * 70)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "cleaned_sms_dataset.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "svm_model.pkl"
)


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128

BATCH_SIZE = 16

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ------------------------------------------------------------
# 3. REPRODUCIBILITY
# ------------------------------------------------------------

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)


# ------------------------------------------------------------
# 4. DEVICE
# ------------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ------------------------------------------------------------
# 5. CHECK DATASET
# ------------------------------------------------------------

print("\n[1/8] Checking dataset...")

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )

print(
    "Dataset found:",
    DATA_FILE
)


# ------------------------------------------------------------
# 6. LOAD DATASET
# ------------------------------------------------------------

print("\n[2/8] Loading cleaned dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    "Dataset shape:",
    df.shape
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ------------------------------------------------------------
# 7. VALIDATE DATA
# ------------------------------------------------------------

required_columns = [
    "message",
    "target"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


df = df.dropna(
    subset=[
        "message",
        "target"
    ]
).copy()


df["message"] = (
    df["message"]
    .astype(str)
)


df["target"] = (
    df["target"]
    .astype(int)
)


print("\nClass distribution:")

print(
    df["target"]
    .value_counts()
)


# ------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\n[3/8] Creating train/test split...")

X = df["message"].astype(str).to_numpy()

y = df["target"].astype(int).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ------------------------------------------------------------
# 9. LOAD TRANSFORMER
# ------------------------------------------------------------

print("\n[4/8] Loading DistilBERT Transformer...")

print(
    "Model:",
    MODEL_NAME
)

print(
    "The model may download the first time."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


transformer = AutoModel.from_pretrained(
    MODEL_NAME
)


transformer = transformer.to(
    DEVICE
)


transformer.eval()


print(
    "Transformer loaded successfully."
)


# ------------------------------------------------------------
# 10. MEAN POOLING
# ------------------------------------------------------------

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )

    sum_embeddings = torch.sum(
        token_embeddings
        * input_mask_expanded,
        dim=1
    )

    sum_mask = torch.clamp(
        input_mask_expanded.sum(
            dim=1
        ),
        min=1e-9
    )

    return (
        sum_embeddings
        / sum_mask
    )


# ------------------------------------------------------------
# 11. EMBEDDING FUNCTION
# ------------------------------------------------------------

def create_embeddings(
    texts,
    batch_size=BATCH_SIZE
):

    embeddings = []

    total = len(texts)

    print(
        f"Creating embeddings for "
        f"{total} messages..."
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = list(
            texts[start:end]
        )

        encoded = tokenizer(

            batch,

            padding=True,

            truncation=True,

            max_length=MAX_LENGTH,

            return_tensors="pt"
        )


        encoded = {
            key: value.to(DEVICE)

            for key, value
            in encoded.items()
        }


        with torch.inference_mode():

            outputs = transformer(
                **encoded
            )


            batch_embeddings = (
                mean_pooling(
                    outputs,
                    encoded["attention_mask"]
                )
            )


        embeddings.append(
            batch_embeddings
            .cpu()
            .numpy()
        )


        processed = end

        percentage = (
            processed / total * 100
        )


        print(
            f"\rProcessed: "
            f"{processed}/{total} "
            f"({percentage:.1f}%)",
            end=""
        )


    print()

    return np.vstack(
        embeddings
    )


# ------------------------------------------------------------
# 12. CREATE TRAIN EMBEDDINGS
# ------------------------------------------------------------

print(
    "\n[5/8] Creating training embeddings..."
)

X_train_embeddings = create_embeddings(
    X_train
)


print(
    "\nTraining embedding shape:",
    X_train_embeddings.shape
)


# ------------------------------------------------------------
# 13. CREATE TEST EMBEDDINGS
# ------------------------------------------------------------

print(
    "\n[6/8] Creating testing embeddings..."
)

X_test_embeddings = create_embeddings(
    X_test
)


print(
    "\nTesting embedding shape:",
    X_test_embeddings.shape
)


# ------------------------------------------------------------
# 14. TRAIN LINEAR SVM
# ------------------------------------------------------------

print(
    "\n[7/8] Training Linear SVM..."
)


svm_model = LinearSVC(

    class_weight="balanced",

    random_state=RANDOM_STATE,

    max_iter=5000
)


svm_model.fit(

    X_train_embeddings,

    y_train
)


print(
    "Linear SVM training completed."
)


# ------------------------------------------------------------
# 15. EVALUATION
# ------------------------------------------------------------

print(
    "\nEvaluating model..."
)


y_pred = svm_model.predict(
    X_test_embeddings
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 70)

print(
    "                    MODEL RESULTS"
)

print("=" * 70)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "SAFE",
            "FRAUD"
        ]
    )
)


print(
    "\nConfusion Matrix:"
)


cm = confusion_matrix(
    y_test,
    y_pred
)


print(cm)


# ------------------------------------------------------------
# 16. SAVE MODEL
# ------------------------------------------------------------

print(
    "\n[8/8] Saving trained SVM..."
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


model_data = {

    "model": svm_model,

    "model_name": MODEL_NAME,

    "max_length": MAX_LENGTH,

    "label_mapping": {

        0: "SAFE",

        1: "FRAUD"

    }

}


joblib.dump(
    model_data,
    MODEL_FILE
)


print(
    "\nModel saved successfully:"
)

print(
    MODEL_FILE
)


print("\n")
print("=" * 70)

print(
    "              TRAINING COMPLETED"
)

print("=" * 70)


print(
    "\nYou can now start the Streamlit application."
)

print(
    "\nRun:"
)

print(
    "streamlit run app.py"
)

print()