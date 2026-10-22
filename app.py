import streamlit as st

from predict import predict_sms


# ============================================================
# SMS FRAUD DETECTION
# STREAMLIT WEB APPLICATION
# ============================================================


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="SMS Fraud Detection",
    page_icon="📱",
    layout="centered"
)


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown(
    """
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666666;
    margin-bottom: 30px;
}

.safe-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #eaf7ea;
    border: 1px solid #b7dfb7;
    text-align: center;
    margin-top: 20px;
}

.fraud-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #fdeaea;
    border: 1px solid #e0b4b4;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-description {
    font-size: 17px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #f5f5f5;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #888888;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""",
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">📱 SMS Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered SMS classification using DistilBERT and Linear SVM'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# INFORMATION
# ------------------------------------------------------------

with st.expander("ℹ️ About this application"):

    st.write(
        """
        This application analyzes an SMS message and
        classifies it as either SAFE or FRAUD.

        The machine-learning pipeline uses:

        • DistilBERT Transformer for text representation

        • Numerical embeddings extracted from the Transformer

        • Linear Support Vector Machine (SVM) for classification

        The application was trained using an SMS spam/ham
        dataset, where spam messages are treated as the
        FRAUD class for this project.
        """
    )


# ------------------------------------------------------------
# SMS INPUT
# ------------------------------------------------------------

st.subheader("Enter an SMS message")

message = st.text_area(
    "SMS Message",
    height=180,
    placeholder=(
        "Example:\n\n"
        "Congratulations! You have won a prize. "
        "Click the link to claim your reward."
    ),
    label_visibility="collapsed"
)


# ------------------------------------------------------------
# EXAMPLE MESSAGES
# ------------------------------------------------------------

st.markdown("**Try an example:**")

example1, example2 = st.columns(2)


with example1:

    if st.button(
        "🚨 Fraud Example",
        use_container_width=True
    ):

        message = (
            "Congratulations! You have won "
            "a £1000 prize. Call now to claim "
            "your reward."
        )


with example2:

    if st.button(
        "✅ Safe Example",
        use_container_width=True
    ):

        message = (
            "Hey, are we still meeting for "
            "lunch at 1 pm today?"
        )


# ------------------------------------------------------------
# ANALYZE BUTTON
# ------------------------------------------------------------

st.markdown("")

analyze = st.button(
    "🔍 Analyze SMS",
    type="primary",
    use_container_width=True
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if analyze:

    if not message.strip():

        st.warning(
            "Please enter an SMS message first."
        )

    else:

        with st.spinner("Analyzing SMS..."):

            try:

                result = predict_sms(message)

                label = result["label"]

                decision_score = result["decision_score"]


                # ------------------------------------------------
                # FRAUD RESULT
                # ------------------------------------------------

                if label == "FRAUD":

                    st.markdown(
                        """
<div class="fraud-box">
    <div class="result-title">🚨 FRAUD</div>
    <div class="result-description">
        This SMS has been classified as suspicious.
    </div>
</div>
""",
                        unsafe_allow_html=True
                    )

                    st.error(
                        "⚠️ Be careful with this message. "
                        "Avoid clicking unknown links or "
                        "sharing personal information."
                    )


                # ------------------------------------------------
                # SAFE RESULT
                # ------------------------------------------------

                else:

                    st.markdown(
                        """
<div class="safe-box">
    <div class="result-title">✅ SAFE</div>
    <div class="result-description">
        This SMS has been classified as safe by the model.
    </div>
</div>
""",
                        unsafe_allow_html=True
                    )

                    st.success(
                        "The message does not appear "
                        "suspicious according to the model."
                    )


                # ------------------------------------------------
                # MODEL INFORMATION
                # ------------------------------------------------

                st.markdown(
                    f"""
<div class="info-box">
    <b>Model:</b> DistilBERT + Linear SVM
    <br><br>
    <b>Decision score:</b> {decision_score:.4f}
</div>
""",
                    unsafe_allow_html=True
                )

                st.caption(
                    "Note: The decision score is the "
                    "SVM decision-function value, not a "
                    "calibrated probability."
                )


                # ------------------------------------------------
                # MESSAGE DISPLAY
                # ------------------------------------------------

                st.subheader("Analyzed message")

                st.info(message)


            except Exception as e:

                st.error(
                    "An error occurred while analyzing "
                    "the message."
                )

                st.exception(e)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown(
    """
<div class="footer">

SMS Fraud Detection System<br>
DistilBERT Transformer + Linear SVM

</div>
""",
    unsafe_allow_html=True
)