import streamlit as st
import os
from features import load_svm_model_and_vectorizer
from preprocess import preprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

st.set_page_config(page_title="Review Sentiment Analysis", layout="centered")

# Dynamic gradient background: red, black, and bright white
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #E50914 0%, #221f1f 60%, #f5f5f1 100%) !important;
    min-height: 100vh;
    min-width: 100vw;
    margin: 0;
    padding: 0;
}
.block-container {
    max-width: 800px !important;
    margin: 3rem auto 3rem auto !important;
    background: rgba(255,255,255,0.90);
    border-radius: 18px;
    padding: 2.5rem 2rem 2rem 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    color: #111 !important;
    font-weight: bold !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    font-weight: bold !important;
    font-size: 2.2rem !important;
    text-align: center !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    color: #111 !important;
}
.stMarkdown, .stText, .stDataFrame, .stButton, .stFileUploader, .stTextArea {
    font-size: 1.15rem !important;
    font-weight: bold !important;
    color: #111 !important;
}
.stTextArea textarea {
    min-height: 300px !important;
    font-size: 1.1rem !important;
    border-radius: 10px !important;
    margin-bottom: 1.5rem !important;
    color: #111 !important;
    font-weight: bold !important;
}
.stButton button, .stDownloadButton button {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    padding: 0.6em 1.5em !important;
    margin-top: 1em !important;
    color: #fff !important;
    background: #E50914 !important;
    border: none !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #b0060f !important;
    color: #fff !important;
}
.stDataFrame, .stTable {
    margin-top: 1.5em !important;
    margin-bottom: 1.5em !important;
    color: #111 !important;
    font-weight: bold !important;
}
/* Ensure Streamlit message boxes (warning, error, info, success) are readable */
.stAlert, .stAlert p, .stAlert span, .stAlert div {
    color: #111 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:1.3rem; font-weight:bold; margin-bottom:2rem; color:#111;'>Predict the sentiment of reviews or upload a CSV for batch analysis.</div>", unsafe_allow_html=True)

# Check for SVM model
if not (os.path.exists("svm_sentiment_model.joblib") and os.path.exists("tfidf_vectorizer.joblib")):
    st.error("No trained SVM model found. Please run training script first.")
    st.stop()

model, vectorizer = load_svm_model_and_vectorizer()

# --- Multi-Review Prediction ---
st.markdown("<label style='color:#111; font-size:1.15rem; font-weight:bold;'>Enter one or more reviews (one per line):</label>", unsafe_allow_html=True)
reviews_input = st.text_area(
    label=" ",  # Hide default label
    height=300,
    max_chars=8000,
    placeholder="Type or paste reviews here, each on a new line..."
)
predict_btn = st.button("Predict")

final_outcome_html = ""

if predict_btn and reviews_input.strip():
    reviews = [r.strip() for r in reviews_input.split('\n') if r.strip()]
    if not reviews:
        st.warning("Please enter at least one review.")
    else:
        clean_reviews = [preprocess(r) for r in reviews]
        X = vectorizer.transform(clean_reviews)
        preds = model.predict(X)
        confs = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:,1]
        confs = [abs(c) if hasattr(model, 'decision_function') else c for c in confs]
        result_df = pd.DataFrame({
            'review': reviews,
            'sentiment': preds,
            'confidence': [f"{c:.2f}" for c in confs]
        })
        st.dataframe(result_df, use_container_width=True)
        # Final outcome summary
        total = len(result_df)
        pos = (result_df['sentiment'] == 'positive').sum()
        neg = (result_df['sentiment'] == 'negative').sum()
        st.markdown(f"<div style='margin-top:1em; padding:1em; border-radius:8px; background:#f4f6fb; display:inline-block; font-size:1.15rem; font-weight:bold; color:#111;'><b>Final Outcome:</b><br>Total reviews: <b>{total}</b><br>Positive: <b>{pos}</b> ({(pos/total*100):.1f}%)<br>Negative: <b>{neg}</b> ({(neg/total*100):.1f}%)</div>", unsafe_allow_html=True)
        # Visually distinct final output
        if pos > neg:
            final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#a8ff78 0%,#78ffd6 100%); color:#155724; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Most reviews are <b>Positive</b>! 🎉</div>"
        elif neg > pos:
            final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#ffecd2 0%,#fcb69f 100%); color:#721c24; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Most reviews are <b>Negative</b>.</div>"
        else:
            final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#e0eafc 0%,#cfdef3 100%); color:#333; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Equal number of positive and negative reviews.</div>"
        st.markdown(final_outcome_html, unsafe_allow_html=True)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "sentiment_predictions.csv", "text/csv")

# --- Batch Prediction via CSV ---
st.markdown("<label style='color:#111; font-size:1.15rem; font-weight:bold;'>Upload a CSV file with a 'review' column</label>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            df['clean_review'] = df['review'].astype(str).apply(preprocess)
            X = vectorizer.transform(df['clean_review'])
            preds = model.predict(X)
            confs = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:,1]
            confs = [abs(c) if hasattr(model, 'decision_function') else c for c in confs]
            result_df = pd.DataFrame({
                'review': df['review'],
                'sentiment': preds,
                'confidence': [f"{c:.2f}" for c in confs]
            })
            st.dataframe(result_df, use_container_width=True)
            # Final outcome summary
            total = len(result_df)
            pos = (result_df['sentiment'] == 'positive').sum()
            neg = (result_df['sentiment'] == 'negative').sum()
            st.markdown(f"<div style='margin-top:1em; padding:1em; border-radius:8px; background:#f4f6fb; display:inline-block; font-size:1.15rem; font-weight:bold; color:#111;'><b>Final Outcome:</b><br>Total reviews: <b>{total}</b><br>Positive: <b>{pos}</b> ({(pos/total*100):.1f}%)<br>Negative: <b>{neg}</b> ({(neg/total*100):.1f}%)</div>", unsafe_allow_html=True)
            # Visually distinct final output
            if pos > neg:
                final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#a8ff78 0%,#78ffd6 100%); color:#155724; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Most reviews are <b>Positive</b>! 🎉</div>"
            elif neg > pos:
                final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#ffecd2 0%,#fcb69f 100%); color:#721c24; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Most reviews are <b>Negative</b>.</div>"
            else:
                final_outcome_html = f"<div style='margin-top:2em; padding:1.5em; border-radius:12px; background:linear-gradient(90deg,#e0eafc 0%,#cfdef3 100%); color:#333; font-size:1.5em; text-align:center; font-weight:bold;'><b>Final Output:</b><br>Equal number of positive and negative reviews.</div>"
            st.markdown(final_outcome_html, unsafe_allow_html=True)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "sentiment_predictions.csv", "text/csv")

            # --- SVM Analysis on Uploaded Data ---
            st.markdown("<h2 style='font-size:1.3rem; font-weight:bold; color:#111;'>SVM Analysis on Uploaded Data</h2>", unsafe_allow_html=True)
            # 1. Sentiment Distribution
            fig1, ax1 = plt.subplots()
            sns.countplot(x='sentiment', data=result_df, palette='pastel', ax=ax1)
            ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold', color='#111')
            st.pyplot(fig1)

            # 2. Feature Importance (Top 10)
            if hasattr(model, 'coef_'):
                st.markdown("<b style='color:#111;'>Top 10 Important Features (words):</b>", unsafe_allow_html=True)
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                top_pos_idx = np.argsort(coefs)[-10:][::-1]
                top_neg_idx = np.argsort(coefs)[:10]
                top_features = [(feature_names[i], coefs[i]) for i in top_pos_idx] + [(feature_names[i], coefs[i]) for i in top_neg_idx]
                feat_df = pd.DataFrame(top_features, columns=['feature', 'coefficient'])
                fig2, ax2 = plt.subplots()
                sns.barplot(x='coefficient', y='feature', data=feat_df, palette='coolwarm', ax=ax2)
                ax2.set_title('Top 10 Positive & Negative Features', fontsize=14, fontweight='bold', color='#111')
                st.pyplot(fig2)

            # 3. Confusion Matrix (if true labels available)
            if 'sentiment_true' in df.columns:
                st.markdown("<b style='color:#111;'>Confusion Matrix (vs. true labels):</b>", unsafe_allow_html=True)
                cm = confusion_matrix(df['sentiment_true'], preds, labels=['positive', 'negative'])
                fig3, ax3 = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'], ax=ax3)
                ax3.set_xlabel('Predicted', fontsize=12, fontweight='bold', color='#111')
                ax3.set_ylabel('True', fontsize=12, fontweight='bold', color='#111')
                ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#111')
                st.pyplot(fig3)
    except Exception as e:
        st.error(f"Error processing file: {e}") 