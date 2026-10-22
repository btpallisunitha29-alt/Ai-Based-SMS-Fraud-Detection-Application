import pandas as pd
import re
from pathlib import Path


# ============================================================
# SMS FRAUD DETECTION
# DATA PREPROCESSING
# ============================================================

print("=" * 60)
print("        SMS FRAUD DETECTION - DATA PREPROCESSING")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "sms_dataset.csv.xlsx"

DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = DATA_DIR / "cleaned_sms_dataset.csv"


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

print("\n[1/9] Checking dataset...")

if not INPUT_FILE.exists():
    print("\nERROR: Dataset not found!")
    print(f"Expected location:\n{INPUT_FILE}")
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

print(f"Dataset found: {INPUT_FILE}")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n[2/9] Loading dataset...")

df_raw = pd.read_excel(
    INPUT_FILE,
    header=None
)

print(f"Raw dataset shape: {df_raw.shape}")

print("\nFirst 5 raw rows:")
print(df_raw.head().to_string(index=False))


# ============================================================
# 4. EXTRACT LABEL AND MESSAGE
# ============================================================

print("\n[3/9] Extracting labels and messages...")

# The actual Excel file contains:
#
# Column 0:
# ham     message
# spam    message
#
# Column 1 is empty.
#
# We therefore use the first column and split:
# ham/spam + message

raw_text = (
    df_raw.iloc[:, 0]
    .astype(str)
    .str.strip()
)


# Extract label and complete message
extracted = raw_text.str.extract(
    r"^\s*(ham|spam)\s+(.*)$",
    flags=re.IGNORECASE,
    expand=True
)


extracted.columns = [
    "label",
    "message"
]


df = extracted.copy()


print("\nExtracted columns:")
print(df.columns.tolist())


# ============================================================
# 5. CLEAN DATA
# ============================================================

print("\n[4/9] Cleaning label and message columns...")

original_rows = len(df)


# Remove rows where extraction failed
before_drop = len(df)

df = df.dropna(
    subset=[
        "label",
        "message"
    ]
).copy()

extraction_failed = (
    before_drop - len(df)
)


print(
    f"Rows that could not be extracted: "
    f"{extraction_failed}"
)


# Clean labels
df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# Clean messages
def clean_sms(text):

    text = str(text)

    # Replace multiple spaces/newlines
    # with a single space
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["message"] = (
    df["message"]
    .apply(clean_sms)
)


# ============================================================
# 6. VALIDATE LABELS
# ============================================================

print("\n[5/9] Validating labels...")

valid_labels = {
    "ham",
    "spam"
}


invalid_labels = ~df["label"].isin(
    valid_labels
)


invalid_count = invalid_labels.sum()


if invalid_count > 0:

    print(
        f"Invalid labels removed: "
        f"{invalid_count}"
    )

    df = df[
        ~invalid_labels
    ].copy()


print("\nLabel distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# REMOVE EMPTY MESSAGES
# ============================================================

empty_messages = (
    df["message"]
    .str.strip()
    .eq("")
)


empty_count = empty_messages.sum()


if empty_count > 0:

    print(
        f"Empty messages removed: "
        f"{empty_count}"
    )

    df = df[
        ~empty_messages
    ].copy()


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

print("\n[6/9] Removing duplicate messages...")

before_duplicates = len(df)


df = df.drop_duplicates(
    subset=[
        "label",
        "message"
    ],
    keep="first"
).copy()


duplicates_removed = (
    before_duplicates - len(df)
)


print(
    f"Duplicate records removed: "
    f"{duplicates_removed}"
)


# ============================================================
# 8. CREATE TARGET LABELS
# ============================================================

print("\n[7/9] Creating target labels...")

# ham  = SAFE  = 0
# spam = FRAUD = 1

df["target"] = df["label"].map(
    {
        "ham": 0,
        "spam": 1
    }
)


if df["target"].isnull().any():

    raise ValueError(
        "Some records could not be converted "
        "to target labels."
    )


df["target"] = (
    df["target"]
    .astype(int)
)


# Reset index
df = df.reset_index(
    drop=True
)


# ============================================================
# 9. SAVE CLEANED DATASET
# ============================================================

print("\n[8/9] Saving cleaned dataset...")

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print(
    f"\nCleaned dataset saved to:"
)

print(OUTPUT_FILE)


# ============================================================
# DATASET STATISTICS
# ============================================================

print("\n[9/9] Generating dataset statistics...")


total_records = len(df)

ham_count = (
    df["label"]
    .eq("ham")
    .sum()
)

spam_count = (
    df["label"]
    .eq("spam")
    .sum()
)


if total_records > 0:

    ham_percentage = (
        ham_count / total_records * 100
    )

    spam_percentage = (
        spam_count / total_records * 100
    )

    average_length = (
        df["message"]
        .str.len()
        .mean()
    )

    minimum_length = (
        df["message"]
        .str.len()
        .min()
    )

    maximum_length = (
        df["message"]
        .str.len()
        .max()
    )

else:

    ham_percentage = 0
    spam_percentage = 0
    average_length = 0
    minimum_length = 0
    maximum_length = 0


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("             DATA CLEANING COMPLETED")
print("=" * 60)


print(
    f"\nOriginal records       : "
    f"{original_rows}"
)

print(
    f"Final records          : "
    f"{total_records}"
)


print(
    f"\nSAFE (ham)             : "
    f"{ham_count}"
)

print(
    f"FRAUD (spam)           : "
    f"{spam_count}"
)


print(
    f"\nSAFE percentage        : "
    f"{ham_percentage:.2f}%"
)

print(
    f"FRAUD percentage       : "
    f"{spam_percentage:.2f}%"
)


print(
    f"\nDuplicates removed     : "
    f"{duplicates_removed}"
)


print(
    f"Average SMS length     : "
    f"{average_length:.2f} characters"
)

print(
    f"Shortest SMS           : "
    f"{minimum_length} characters"
)

print(
    f"Longest SMS            : "
    f"{maximum_length} characters"
)


print("\nFinal columns:")

print(
    df.columns.tolist()
)


print("\nFirst 5 cleaned records:")

print(
    df.head(5).to_string(
        index=False
    )
)


print("\nOutput file:")

print(OUTPUT_FILE)


print("\n")
print("=" * 60)
print("              PREPROCESSING SUCCESSFUL")
print("=" * 60)

print(
    "\nYour cleaned dataset is ready "
    "for the Transformer."
)