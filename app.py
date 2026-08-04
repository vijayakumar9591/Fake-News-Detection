"""
Streamlit Web Application for Fake News Detection.

This interactive dashboard allows users to enter or select news articles,
preprocess text in real time, generate model predictions using saved Joblib models,
and view confidence scores, probability breakdowns, and benchmark metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Import preprocessing helper
from preprocess import clean_text

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="TruthLens - AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark UI Styling
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Header Banner */
    .header-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        text-align: center;
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Custom Cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }

    /* Result Badges */
    .result-badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
    }

    .result-badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.4);
    }

    /* Metric Values */
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """
    Caches and loads the trained model, vectorizer, and evaluation metrics.
    
    Returns:
        tuple: (model, vectorizer, metrics_dict)
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return model, vectorizer, metrics


# Pre-defined sample news for instant testing
SAMPLE_NEWS = {
    "Select a sample...": "",
    "Real News: NASA Spacecraft Mission": (
        "WASHINGTON (Reuters) - NASA mission controllers confirmed today that the OSIRIS-REx spacecraft "
        "has successfully collected rock and dust samples from the surface of asteroid Bennu. The spacecraft "
        "navigated its robotic arm to contact the asteroid surface for approximately six seconds before returning "
        "to a safe orbit. Scientists expect the pristine sample to provide crucial insights into early solar system "
        "formation and organic compounds."
    ),
    "Real News: Central Bank Interest Rate": (
        "BRUSSELS (Reuters) - European financial regulators voted today to adjust key monetary benchmark interest rates "
        "by 25 basis points in response to stabilizing inflation indices across euro-area economies. Economic "
        "analysts noted that equity markets responded positively to the announced economic policy guidelines."
    ),
    "Fake News: Secret Alien Machine": (
        "SHOCKING BREAKING NEWS: Anonymous whistleblowers inside secret government installations have revealed "
        "that top-secret high-frequency scalar cannons are secretly controlling global weather events! Mainstream "
        "media sources refuse to report on this terrifying secret. Watch this viral video before authorities delete it!"
    ),
    "Fake News: Miracle Juice Cure": (
        "UNBELIEVABLE DISCOVERY: A rogue doctor who was banned by pharmaceutical conglomerates claims that a secret "
        "mixture of cucumber juice and ancient crystals cures all known human diseases instantly! Doctors hate this "
        "simple home trick. Click here to order your secret bottle now!"
    )
}


def main():
    # Sidebar Navigation & Model Info
    with st.sidebar:
        st.title("🛡️ TruthLens AI")
        st.caption("Machine Learning Fake News Detection System")
        st.markdown("---")

        st.subheader("🤖 Model Configuration")
        model, vectorizer, metrics = load_model_artifacts()

        if model is not None:
            best_name = metrics.get("best_model", "Logistic Regression") if metrics else "Machine Learning Model"
            st.success(f"**Loaded Model:** {best_name}")
            st.info("**Vectorizer:** TF-IDF (5000 max features, n-grams 1-2)")
            st.write("---")

            if metrics and "models" in metrics:
                st.subheader("📊 Model Benchmarks")
                for m_name, m_stats in metrics["models"].items():
                    st.write(f"**{m_name}**")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Accuracy", f"{m_stats['accuracy']*100:.1f}%")
                    col_b.metric("F1-Score", f"{m_stats['f1_score']*100:.1f}%")
        else:
            st.error("Model artifacts not found! Please run `python train_model.py` first.")
            st.info("Run `python train_model.py` in your terminal to build `models/best_model.joblib`.")

        st.markdown("---")
        st.markdown(
            "**Built with:**\n"
            "- Python & Streamlit\n"
            "- NLTK (Stemming & Stopwords)\n"
            "- Scikit-Learn (TF-IDF & Classifiers)\n"
            "- Joblib Serialization"
        )

    # Main Header
    st.markdown(
        """
        <div class="header-container">
            <div class="header-title">🔍 Fake News Detector</div>
            <div class="header-subtitle">
                Analyze news articles instantly with Natural Language Processing and Machine Learning models. 
                Identify misinformation, assess credibility, and inspect probability distributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Predict Article", "📊 Model Comparison", "ℹ️ How it Works"])

    # TAB 1: PREDICTION ENGINE
    with tab1:
        st.subheader("Enter News Text for Verification")

        # Preset sample loader dropdown
        selected_sample = st.selectbox(
            "💡 Need an example article? Choose from preset samples:",
            list(SAMPLE_NEWS.keys())
        )

        initial_text = SAMPLE_NEWS[selected_sample] if selected_sample != "Select a sample..." else ""

        # Text input area
        user_input = st.text_area(
            "Paste news title and main content below:",
            value=initial_text,
            height=200,
            placeholder="Type or paste the news headline and body text here..."
        )

        col_count1, col_count2, col_btn = st.columns([2, 2, 2])

        word_count = len(user_input.split()) if user_input else 0
        char_count = len(user_input) if user_input else 0

        with col_count1:
            st.caption(f"📏 **Word Count:** {word_count} words")
        with col_count2:
            st.caption(f"🔤 **Character Count:** {char_count} characters")

        with col_btn:
            predict_clicked = st.button("🚀 Analyze Credibility", type="primary", use_container_width=True)

        if predict_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter news text or select a preset sample article above!")
            elif model is None or vectorizer is None:
                st.error("❌ Model artifacts are missing. Run `python train_model.py` to train the models.")
            else:
                with st.spinner("Cleaning text, extracting TF-IDF features, and running predictions..."):
                    # Step 1: Preprocess raw input
                    cleaned_input = clean_text(user_input)

                    # Step 2: Vectorize preprocessed text
                    input_tfidf = vectorizer.transform([cleaned_input])

                    # Step 3: Predict class label and probabilities
                    prediction = model.predict(input_tfidf)[0]  # 0 = FAKE, 1 = REAL
                    probabilities = model.predict_proba(input_tfidf)[0]

                    prob_fake = float(probabilities[0])
                    prob_real = float(probabilities[1])

                    confidence = prob_real if prediction == 1 else prob_fake

                st.markdown("---")
                st.subheader("🎯 Prediction Result & Confidence Analysis")

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if prediction == 1:
                        st.markdown(
                            """
                            <div class="result-badge-real">
                                ✅ REAL NEWS
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.success(f"**Confidence Score:** {confidence * 100:.2f}% probability of being authentic.")
                    else:
                        st.markdown(
                            """
                            <div class="result-badge-fake">
                                🚨 FAKE NEWS
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.error(f"**Confidence Score:** {confidence * 100:.2f}% probability of being fake/misinformation.")

                with res_col2:
                    # Plotly Probability Bar Chart
                    fig = go.Figure(go.Bar(
                        x=[prob_fake * 100, prob_real * 100],
                        y=['FAKE', 'REAL'],
                        orientation='h',
                        marker=dict(
                            color=['#ef4444', '#10b981'],
                            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                        ),
                        text=[f"{prob_fake*100:.1f}%", f"{prob_real*100:.1f}%"],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title="Probability Distribution (%)",
                        xaxis=dict(range=[0, 100], title="Probability Score (%)"),
                        yaxis=dict(title="Prediction Class"),
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f8fafc')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Preprocessing & Extracted Key Features Breakdown
                st.markdown("---")
                st.subheader("🔍 NLP Insights & Extracted Top Features")

                exp_col1, exp_col2 = st.columns(2)

                with exp_col1:
                    st.markdown("**Cleaned & Stemmed Text Sample:**")
                    st.code(cleaned_input if cleaned_input else "[No features extracted]", language="text")

                with exp_col2:
                    st.markdown("**Top Extracted TF-IDF Keywords:**")
                    feature_names = np.array(vectorizer.get_feature_names_out())
                    dense_vec = input_tfidf.toarray()[0]
                    nonzero_indices = dense_vec.nonzero()[0]

                    if len(nonzero_indices) > 0:
                        top_indices = nonzero_indices[np.argsort(dense_vec[nonzero_indices])[-6:][::-1]]
                        top_features = pd.DataFrame({
                            'Keyword': feature_names[top_indices],
                            'TF-IDF Weight': dense_vec[top_indices]
                        })
                        st.dataframe(top_features, use_container_width=True, hide_index=True)
                    else:
                        st.info("No recognizable keywords matched the vocabulary.")

    # TAB 2: MODEL COMPARISON BENCHMARKS
    with tab2:
        st.subheader("📈 Model Evaluation & Accuracy Comparison")

        if metrics and "models" in metrics:
            m_df = []
            for name, stats in metrics["models"].items():
                m_df.append({
                    "Model": name,
                    "Accuracy": stats["accuracy"],
                    "Precision": stats["precision"],
                    "Recall": stats["recall"],
                    "F1 Score": stats["f1_score"]
                })

            df_metrics = pd.DataFrame(m_df)

            st.dataframe(
                df_metrics.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Grouped Bar Chart of Metrics Comparison
            fig_compare = px.bar(
                df_metrics,
                x="Model",
                y=["Accuracy", "Precision", "Recall", "F1 Score"],
                barmode="group",
                title="Model Performance Metric Comparison",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#10b981"]
            )

            fig_compare.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                yaxis=dict(range=[0.5, 1.05], title="Score (0.0 to 1.0)"),
                height=400
            )

            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("Run `python train_model.py` to generate model benchmark comparison reports.")

    # TAB 3: HOW IT WORKS & SYSTEM ARCHITECTURE
    with tab3:
        st.subheader("🧠 Machine Learning Pipeline Architecture")

        st.markdown(
            """
            ### 1. Data Cleaning & Text Preprocessing
            - **Lowercasing & Normalization**: Strips URLs, HTML tags, digits, and special characters.
            - **Punctuation Stripping**: Uses Python string translation tables to filter out noise.
            - **Stop Words Removal**: Filters standard English stop words using NLTK `stopwords`.
            - **Stemming**: Applies NLTK `PorterStemmer` to convert words to base linguistic roots (e.g. *'reporting'* ➔ *'report'*).

            ### 2. Feature Extraction (TF-IDF Vectorization)
            - Converts stemmed text into numerical vectors using `TfidfVectorizer`.
            - Computes Term Frequency-Inverse Document Frequency using **unigrams and bigrams** (1-2 word combinations).
            - Vocabulary size limited to top **5,000 max features** to optimize predictive power and speed.

            ### 3. Classification Algorithms
            - **Logistic Regression**: Linear binary classifier optimizing decision boundaries.
            - **Multinomial Naive Bayes**: Probabilistic classifier ideal for word frequency matrices.
            - Model evaluation selects the top algorithm based on **F1-score & Accuracy** on test data.
            """
        )


if __name__ == "__main__":
    main()
