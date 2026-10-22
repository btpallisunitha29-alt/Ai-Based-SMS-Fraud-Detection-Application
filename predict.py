
import torch
import joblib

from pathlib import Path
from transformers import AutoTokenizer, AutoModel


# ============================================================
# SMS FRAUD DETECTION
# PREDICTION ENGINE
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "svm_model.pkl"
)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# CHECK MODEL
# ------------------------------------------------------------

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_FILE}\n\n"
        "Run train.py first."
    )


# ------------------------------------------------------------
# LOAD TRAINED SVM
# ------------------------------------------------------------

model_data = joblib.load(
    MODEL_FILE
)

svm_model = model_data["model"]

MODEL_NAME = model_data["model_name"]

MAX_LENGTH = model_data["max_length"]


# ------------------------------------------------------------
# LOAD DISTILBERT
# ------------------------------------------------------------

print("Loading DistilBERT...")

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

print("DistilBERT loaded successfully.")


# ------------------------------------------------------------
# MEAN POOLING
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
# CREATE SMS EMBEDDING
# ------------------------------------------------------------

def create_embedding(message):

    encoded = tokenizer(

        [message],

        padding=True,

        truncation=True,

        max_length=MAX_LENGTH,

        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)

        for key, value in encoded.items()
    }

    with torch.inference_mode():

        outputs = transformer(
            **encoded
        )

        embedding = mean_pooling(
            outputs,
            encoded["attention_mask"]
        )

    return embedding.cpu().numpy()


# ------------------------------------------------------------
# PREDICT SMS
# ------------------------------------------------------------

# ------------------------------------------------------------
# PREDICT SMS
# ------------------------------------------------------------

def predict_sms(message):

    if not message:

        raise ValueError(
            "SMS message cannot be empty."
        )

    message = str(message).strip()

    if not message:

        raise ValueError(
            "SMS message cannot be empty."
        )


    # Create DistilBERT embedding

    embedding = create_embedding(
        message
    )


    # SVM prediction

    prediction = svm_model.predict(
        embedding
    )[0]


    # SVM decision score

    decision_score = (
        svm_model.decision_function(
            embedding
        )[0]
    )


    # 0 = SAFE
    # 1 = FRAUD

    if prediction == 1:

        label = "FRAUD"

    else:

        label = "SAFE"


    return {

        "label": label,

        "prediction": int(
            prediction
        ),

        "decision_score": float(
            decision_score
        )

    }


# ============================================================
# TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    test_message = (
        "Congratulations! You have won "
        "a £1000 prize. Call now to claim "
        "your reward."
    )


    result = predict_sms(
        test_message
    )


    print("\nPrediction Result:")

    print(result)