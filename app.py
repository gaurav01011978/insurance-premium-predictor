import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Medical Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
    }
    .result-amount {
        font-size: 3rem;
        font-weight: 800;
        color: #ffd700;
    }
    .info-card {
        background: #1e3a5f;
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #5ba3f5;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model         = joblib.load('model/rf_model.pkl')
    scaler        = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.info("Please make sure rf_model.pkl, scaler.pkl and feature_names.pkl are inside the model/ folder.")
    st.stop()


def get_bmi_category(bmi):
    if bmi < 18.5:   return 'Underweight'
    elif bmi < 25:   return 'Normal'
    elif bmi < 30:   return 'Overweight'
    else:            return 'Obese'

def get_age_group(age):
    if age < 30:    return 'Young'
    elif age < 50:  return 'Middle'
    else:           return 'Senior'

ENCODINGS = {
    'sex':          {'male': 1, 'female': 0},
    'smoker':       {'yes': 1, 'no': 0},
    'region':       {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3},
    'bmi_category': {'Normal': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3},
    'age_group':    {'Middle': 0, 'Senior': 1, 'Young': 2},
}

def preprocess_input(age, sex, bmi, children, smoker, region):
    bmi_cat = get_bmi_category(bmi)
    age_grp = get_age_group(age)
    data = {
        'age':          age,
        'sex':          ENCODINGS['sex'][sex],
        'bmi':          bmi,
        'children':     children,
        'smoker':       ENCODINGS['smoker'][smoker],
        'region':       ENCODINGS['region'][region],
        'bmi_category': ENCODINGS['bmi_category'][bmi_cat],
        'age_group':    ENCODINGS['age_group'][age_grp],
    }
    df = pd.DataFrame([data])[feature_names]
    return df, bmi_cat, age_grp


st.markdown('<p class="main-header">🏥 Medical Insurance Premium Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter your details to estimate your annual medical insurance premium</p>', unsafe_allow_html=True)
st.markdown("---")

left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("📋 Enter Your Details")
    col1, col2 = st.columns(2)

    with col1:
        age      = st.slider("🎂 Age", min_value=18, max_value=64, value=30)
        bmi      = st.slider("⚖️ BMI", min_value=10.0, max_value=53.0, value=25.0, step=0.1)
        children = st.selectbox("👶 Number of Dependents", options=[0, 1, 2, 3, 4, 5])

    with col2:
        sex    = st.radio("⚧ Gender", options=['male', 'female'])
        smoker = st.radio("🚬 Do you smoke?", options=['no', 'yes'])
        region = st.selectbox("📍 Region", options=['northeast', 'northwest', 'southeast', 'southwest'])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Calculate My Premium")


with right_col:
    st.subheader("📊 Result")

    if predict_btn:
        input_df, bmi_cat, age_grp = preprocess_input(age, sex, bmi, children, smoker, region)
        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div class="result-box">
            <p style="font-size:1.1rem; margin-bottom:0.5rem;">Estimated Annual Premium</p>
            <p class="result-amount">${prediction:,.0f}</p>
            <p style="font-size:0.9rem; opacity:0.8;">Per Year</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if smoker == 'yes' and bmi >= 30:
            risk_msg   = "⚠️ **High Risk**: Smoking + Obese BMI significantly increases your premium."
            risk_color = "#890707"
        elif smoker == 'yes':
            risk_msg   = "🔶 **Medium-High Risk**: Smoking is a major factor in premium calculation."
            risk_color = "#5C0405"
        elif bmi >= 30:
            risk_msg   = "🔶 **Medium Risk**: Obese BMI increases your premium."
            risk_color = "#062f6d"
        else:
            risk_msg   = "✅ **Low Risk**: Your profile looks healthy!"
            risk_color = "#146f14"

        st.markdown(f"""
        <div style="background:{risk_color}; padding:1rem; border-radius:10px; margin:0.5rem 0;">
            {risk_msg}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📝 Your Profile Summary:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="info-card">🎂 Age Group: <b>{age_grp}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">⚖️ BMI Category: <b>{bmi_cat}</b></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="info-card">🚬 Smoker: <b>{smoker.upper()}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card">👶 Dependents: <b>{children}</b></div>', unsafe_allow_html=True)

    else:
        st.info("👈 Fill in your details on the left and click **'Calculate My Premium'**")
        st.markdown("**🤖 About This Model:**")
        st.markdown("""
        <div class="info-card">🌲 Algorithm: Random Forest Regressor</div>
        <div class="info-card">📈 R² Score: ~0.87 (87% accuracy)</div>
        <div class="info-card">📦 Training Data: 1,338 insurance records</div>
        <div class="info-card">🔬 Features: Age, BMI, Smoking, Region, Gender, Children</div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    Built with ❤️ by <strong>Gaurav Kumar</strong> &nbsp;|&nbsp;
    Python · Scikit-learn · Streamlit &nbsp;|&nbsp;
    Dataset: Kaggle Insurance Dataset
</div>
""", unsafe_allow_html=True)