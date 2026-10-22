# ============================================================
# SMS FRAUD DETECTION
# STEP 1: DistilBERT Feature Extraction
#
# Input  : SMS text
# Output : Numerical features (embeddings)
# ============================================================

import os
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = r"C:\sms_project\data\cleaned_sms_dataset.csv"

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
MAX_LENGTH = 128


# ============================================================
# 2. START
# ============================================================

print("=" * 60)
print("SMS FRAUD DETECTION")
print("DISTILBERT FEATURE EXTRACTION")
print("=" * 60)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    print("\nERROR: Dataset not found!")
    print(DATA_FILE)
    exit()

print("\nDataset found:")
print(DATA_FILE)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 5. FIND MESSAGE COLUMN
# ============================================================

message_column = None

possible_columns = [
    "Message",
    "message",
    "SMS",
    "sms",
    "Text",
    "text"
]

for column in possible_columns:

    if column in df.columns:
        message_column = column
        break


# ============================================================
# 6. FALLBACK
# ============================================================

if message_column is None:

    print("\nMessage column was not found automatically.")

    print("Available columns:")
    print(df.columns.tolist())

    # Assume second column is SMS text
    if len(df.columns) >= 2:

        message_column = df.columns[1]

        print("\nUsing:")
        print("Message column:", message_column)

    else:

        print("\nERROR: SMS column not found.")
        exit()


print("\nSMS column:", message_column)


# ============================================================
# 7. PREPARE SMS TEXT
# ============================================================

df = df.dropna(subset=[message_column])

sms_messages = df[message_column].astype(str).tolist()

print("\nNumber of SMS messages:", len(sms_messages))


# ============================================================
# 8. LOAD DISTILBERT TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING DISTILBERT TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# 9. LOAD DISTILBERT MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING DISTILBERT MODEL")
print("=" * 60)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()

print("DistilBERT loaded successfully.")


# ============================================================
# 10. FUNCTION TO CONVERT SMS INTO NUMERICAL FEATURES
# ============================================================

def create_embeddings(texts):

    all_embeddings = []

    total = len(texts)

    print("\nConverting SMS into numerical features...")

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = texts[start:end]

        # ----------------------------------------------------
        # Convert text into tokens
        # ----------------------------------------------------

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        # Move data to CPU/GPU
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # ----------------------------------------------------
        # DistilBERT processes the SMS
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                **encoded
            )

        # ----------------------------------------------------
        # Get numerical representation
        # ----------------------------------------------------

        token_embeddings = output.last_hidden_state

        attention_mask = encoded["attention_mask"]

        # Create mask
        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        # Sum useful token vectors
        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )

        # Count valid tokens
        summed_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        # Mean pooling
        embeddings = summed / summed_mask

        # Convert Tensor → NumPy
        embeddings = embeddings.cpu().numpy()

        all_embeddings.append(embeddings)

        print(
            f"Processed {end}/{total} SMS messages"
        )

    return np.vstack(all_embeddings)


# ============================================================
# 11. CREATE NUMERICAL FEATURES
# ============================================================

embeddings = create_embeddings(
    sms_messages
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DISTILBERT FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print("\nOriginal SMS count:")
print(len(sms_messages))

print("\nNumerical feature shape:")
print(embeddings.shape)


# ============================================================
# 13. SHOW ONE EXAMPLE
# ============================================================

print("\nExample:")

print("\nOriginal SMS:")
print(sms_messages[0])

print("\nFirst 10 numerical features:")
print(embeddings[0][:10])


# ============================================================
# 14. SAVE NUMERICAL FEATURES
# ============================================================

OUTPUT_FILE = r"C:\sms_project\data\transformer_features.npy"

np.save(
    OUTPUT_FILE,
    embeddings
)

print("\nNumerical features saved to:")
print(OUTPUT_FILE)


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRANSFORMER STEP COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
SMS TEXT
   ↓
DistilBERT
   ↓
Numerical Features
   ↓
transformer_features.npy
""")

print("Next step: Linear SVM")
print("=" * 60)