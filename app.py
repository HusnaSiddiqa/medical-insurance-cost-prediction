"""
Medical Insurance Cost Prediction — Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_or_train_model():
    """Load saved model, or train one from scratch if not found (for cloud deployment)."""
    if os.path.exists("models/best_model.pkl"):
        return joblib.load("models/best_model.pkl")

    # Train from scratch using the dataset
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/insurance.csv")
    df["sex"]    = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
    df["region"] = df["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(int)
    df["age_group"]    = pd.cut(df["age"], bins=[0, 25, 40, 55, 100],        labels=[0, 1, 2, 3]).astype(int)
    df["bmi_smoker"]   = df["bmi"] * df["smoker"]

    X = df.drop(columns=["charges"])
    y = df["charges"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    return model


FEATURE_COLS = ["age", "sex", "bmi", "children", "smoker", "region",
                "bmi_category", "age_group", "bmi_smoker"]

def predict(model, age, sex, bmi, children, smoker, region):
    sex_enc    = 1 if sex == "Female" else 0
    smoker_enc = 1 if smoker == "Yes" else 0
    region_enc = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}[region]
    bmi_cat    = 0 if bmi < 18.5 else 1 if bmi < 24.9 else 2 if bmi < 29.9 else 3
    age_grp    = 0 if age < 25 else 1 if age < 40 else 2 if age < 55 else 3
    bmi_smoker = bmi * smoker_enc

    features = pd.DataFrame(
        [[age, sex_enc, bmi, children, smoker_enc, region_enc, bmi_cat, age_grp, bmi_smoker]],
        columns=FEATURE_COLS
    )
    return max(0, model.predict(features)[0])


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Estimate your **annual medical insurance charges** based on your health profile.")
st.markdown("---")

with st.spinner("Loading model..."):
    model = load_or_train_model()

col1, col2 = st.columns(2)

with col1:
    age      = st.slider("Age", 18, 64, 30)
    sex      = st.selectbox("Sex", ["Male", "Female"])
    bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                               help="Body Mass Index — weight(kg) / height(m)²")

with col2:
    children = st.selectbox("Number of Children / Dependents", [0, 1, 2, 3, 4, 5])
    smoker   = st.selectbox("Smoker", ["No", "Yes"])
    region   = st.selectbox("US Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.markdown("---")

bmi_label = ("Underweight" if bmi < 18.5 else
             "Normal"      if bmi < 24.9 else
             "Overweight"  if bmi < 29.9 else "Obese")
bmi_color = ("🟡" if bmi < 18.5 else "🟢" if bmi < 24.9 else "🟠" if bmi < 29.9 else "🔴")
st.info(f"BMI Category: {bmi_color} **{bmi_label}** (BMI: {bmi:.1f})")

if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
    cost = predict(model, age, sex, bmi, children, smoker, region)
    st.success(f"### Estimated Annual Insurance Cost: **${cost:,.2f}**")
    st.caption("This is an ML-based estimate. Actual costs may vary by insurer and policy.")

    with st.expander("Risk Factor Analysis"):
        factors = []
        if smoker == "Yes":
            factors.append("🚨 **Smoker** — Smoking is the #1 cost driver (3–4× premium increase)")
        if bmi >= 30:
            factors.append("⚠️ **Obesity (BMI ≥ 30)** — Significantly raises medical costs")
        if smoker == "Yes" and bmi >= 30:
            factors.append("🔴 **Smoker + Obese** — Highest risk combination; costs increase dramatically")
        if age >= 55:
            factors.append("⚠️ **Senior Age (55+)** — Older patients have higher baseline charges")
        if children >= 3:
            factors.append("ℹ️ **3+ Dependents** — More covered members raise the premium")
        if not factors:
            factors.append("✅ **Low Risk Profile** — Your profile suggests below-average insurance costs")
        for f in factors:
            st.markdown(f)

st.markdown("---")
st.markdown(
    "Built with Python, scikit-learn & Streamlit &nbsp;|&nbsp; "
    "[GitHub](https://github.com/husnasiddiqa/medical-insurance-cost-prediction) &nbsp;|&nbsp; "
    "**Husna Siddiqa**",
    unsafe_allow_html=True
)
