# 🏥 Medical Insurance Cost Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://medical-insurance-cost-prediction-xjmjftdi88a2axuyedabo5.streamlit.app/)

> **End-to-end machine learning project** that predicts individual medical insurance charges using patient demographics and health factors. Includes EDA, feature engineering, multi-model comparison, hyperparameter tuning, and a Streamlit web application for live predictions.

### 🚀 [**Try the Live Demo →**](https://medical-insurance-cost-prediction-xjmjftdi88a2axuyedabo5.streamlit.app/)

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Results](#results)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)
- [Streamlit App](#streamlit-app)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Medical insurance pricing is a complex function of a person's health, lifestyle, and demographics. Insurers use actuarial models to set premiums, but many individuals don't understand what drives their costs. This project builds a transparent, data-driven ML model to **predict annual insurance charges** and identify the most impactful cost drivers.

**Business Value:**
- Helps individuals estimate insurance costs before enrollment
- Assists insurers in risk profiling and premium setting
- Provides explainable predictions that build customer trust

---

## Dataset

| Property | Value |
|----------|-------|
| Records | 1,338 |
| Features | 6 input + 1 target |
| Target | `charges` — annual medical cost in USD |
| Source | [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Age of primary beneficiary (18–64) |
| `sex` | Categorical | Gender (male / female) |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of covered dependents (0–5) |
| `smoker` | Categorical | Smoking status (yes / no) |
| `region` | Categorical | US residential region (NE/NW/SE/SW) |
| `charges` | **Target** | Individual medical costs billed by insurance |

---

## Project Structure

```
medical-insurance-cost-prediction/
│
├── data/
│   └── insurance.csv              # Dataset
│
├── models/                        # Saved model artifacts
│   ├── best_model.pkl             # Trained model (Gradient Boosting)
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_names.json         # Feature list for inference
│   ├── eda_distributions.png      # EDA plots
│   ├── eda_categorical.png
│   ├── eda_feature_impact.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── Medical_Insurance_Cost_Prediction.ipynb   # Main notebook
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ML Pipeline

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
  • Distribution plots (age, BMI, charges)
  • Categorical counts & proportions
  • Feature vs. charges analysis (scatter, boxplots)
  • Correlation heatmap
   │
   ▼
Feature Engineering
  • BMI categories (underweight/normal/overweight/obese)
  • Age groups (young/adult/middle-aged/senior)
  • Smoker × BMI interaction term
  • Label encoding for categorical features
   │
   ▼
Train / Test Split (80% / 20%)
   │
   ▼
Model Training & Evaluation
  ┌────────────────────────────┐
  │  Linear Regression         │
  │  Ridge Regression          │
  │  Lasso Regression          │
  │  Decision Tree             │
  │  Random Forest             │
  │  Gradient Boosting  ◀ Best │
  └────────────────────────────┘
  • 5-Fold Cross-Validation
  • Metrics: R², MAE, RMSE, MAPE
   │
   ▼
Hyperparameter Tuning
  • RandomizedSearchCV on Gradient Boosting
  • 30 iterations, 5-fold CV
   │
   ▼
Model Saving (joblib)
   │
   ▼
Streamlit Deployment
```

---

## Results

### Model Comparison

| Model | Test R² | CV R² | MAE (USD) | RMSE (USD) | MAPE (%) |
|-------|---------|-------|-----------|------------|----------|
| Linear Regression | ~0.82 | ~0.81 | — | — | — |
| Ridge Regression | ~0.82 | ~0.81 | — | — | — |
| Lasso Regression | ~0.82 | ~0.81 | — | — | — |
| Decision Tree | ~0.97 | ~0.96 | — | — | — |
| Random Forest | ~0.99 | ~0.99 | — | — | — |
| **Gradient Boosting** | **0.993** | **~0.99** | **$2,028** | **$2,522** | **7.57%** |

### Best Model: Gradient Boosting Regressor (Tuned)
- **99.3% variance explained** in insurance charges
- **RMSE of $2,522** — predictions within ~$2.5K of actual costs
- **MAPE of 7.57%** — on average 7.6% prediction error
- **Outperforms linear models** due to non-linear feature interactions (especially smoker × BMI)

---

## Key Insights

1. **🚬 Smoking is the #1 Cost Driver** — Smokers pay **3–4× more** than non-smokers. The interaction between smoking and obesity further amplifies costs dramatically.

2. **📈 Age Increases Charges Linearly** — Each additional year of age adds a measurable premium. Older patients have higher baseline medical needs.

3. **⚖️ Obesity Multiplies Smoking Risk** — BMI > 30 alone moderately increases charges, but obese smokers face the highest costs in the dataset.

4. **👨‍👩‍👧 Region Has Minimal Impact** — Geographic region shows little correlation with charges, suggesting national-level pricing is fairly uniform.

5. **🌲 Non-Linear Models Win** — The curved relationship between BMI/age and charges means tree-based models significantly outperform linear regression.

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Medical_Insurance_Cost_Prediction.ipynb
```
Run all cells sequentially. The notebook will:
- Load and explore the dataset
- Generate EDA visualizations
- Train 6 regression models
- Perform hyperparameter tuning
- Save the best model to `models/`

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

---

## Streamlit App

The interactive web app allows real-time insurance cost predictions:

**Features:**
- Sliders and dropdowns for all input features
- Live BMI category indicator (Underweight / Normal / Overweight / Obese)
- Instant cost prediction on button click
- Risk Factor Analysis explaining what drives the predicted cost

**Usage:**
```
1. Adjust the sliders/dropdowns to match patient profile
2. Click "Predict Insurance Cost"
3. Review the estimated annual charge and risk breakdown
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Hyperparameter Tuning** | RandomizedSearchCV |
| **Model Serialization** | joblib |
| **Web App** | Streamlit |
| **Environment** | Jupyter Notebook |

---

## Future Improvements

- [ ] Add XGBoost / LightGBM models for further accuracy gains
- [ ] Integrate SHAP values for model explainability
- [ ] Deploy Streamlit app to Streamlit Cloud or AWS
- [ ] Add confidence intervals to predictions
- [ ] Experiment with log-transformed target variable

---

## Author

**Husna Siddiqa**
- Data Scientist | Machine Learning Engineer
- [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as part of a machine learning portfolio project.*
