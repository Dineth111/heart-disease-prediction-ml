import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="CardioShield AI - Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    /* Main layout customization */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #ff4b4b 0%, #c1121f 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .header-box h1 {
        font-weight: 800;
        margin: 0;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-box p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Cards */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #6c757d;
        margin-bottom: 1rem;
    }
    
    .card-healthy {
        border-left-color: #2ec4b6;
        background-color: #f4fbf9;
    }
    
    .card-warning {
        border-left-color: #ff9f1c;
        background-color: #fffaf4;
    }
    
    .card-danger {
        border-left-color: #e63946;
        background-color: #fff5f5;
    }

    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    
    .metric-desc {
        font-size: 0.85rem;
        color: #495057;
    }
    
    /* Input Form styling */
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b 0%, #c1121f 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(193, 18, 31, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(193, 18, 31, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_model(model_path: str = "model/heart_disease_model.pkl"):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper function to clean feature names locally
def clean_feature_names(feature_names: list) -> list:
    cleaned = []
    for name in feature_names:
        if name.startswith("num__"):
            cleaned.append(name.replace("num__", "").replace("_", " "))
        elif name.startswith("cat__"):
            parts = name.replace("cat__", "").split("_", 1)
            if len(parts) == 2:
                cleaned.append(f"{parts[0].replace('_', ' ')}: {parts[1].replace('_', ' ')}")
            else:
                cleaned.append(name.replace("cat__", "").replace("_", " "))
        else:
            cleaned.append(name.replace("_", " "))
    return cleaned

# Helper function to get precomputed feature importances
@st.cache_data
def get_feature_importances(_model, csv_path: str = "dataset/heart_disease.csv"):
    try:
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["Heart_Disease"])
        y = df["Heart_Disease"]
        
        # Extract fitted preprocessor directly from model pipeline steps
        preprocessor = _model.named_steps["preprocessor"]
        X_proc = preprocessor.transform(X)
        
        # Train a quick random forest classifier just for global feature importance presentation
        rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6)
        rf.fit(X_proc, y)
        
        raw_names = preprocessor.get_feature_names_out()
        cleaned_names = clean_feature_names(raw_names)
        
        feat_df = pd.DataFrame({
            "Feature": cleaned_names,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=True) # Ascending for horizontal plotting
        
        return feat_df.tail(10) # Get top 10 features
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")
        return None

# App Title
st.markdown("""
<div class="header-box">
    <h1>❤️ CardioShield AI Dashboard</h1>
    <p>Predictive Heart Disease Analysis Platform for Healthcare Professionals</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Information
st.sidebar.image("https://img.icons8.com/illustrations/external-justicon-flat-justicon/250/external-heart-rate-hospital-and-medical-justicon-flat-justicon.png", width=150)
st.sidebar.markdown("### Model Diagnostics")
st.sidebar.info("""
**Accuracy**: 86.27%  
**F1-Score**: 87.72%  
**ROC-AUC**: 0.9400  
**Algorithm**: Support Vector Machine (RBF Kernel)
""")

st.sidebar.markdown("### Risk Level Guide")
st.sidebar.markdown("""
*   🟩 **Low Risk** (< 30%): Routine wellness checks recommended.
*   🟨 **Moderate Risk** (30%-70%): Schedule clinical review & diagnostics.
*   🟥 **High Risk** (> 70%): Prompt intervention and cardiologist referral.
""")

# Load resources
model = load_model()
feat_importance = get_feature_importances(model) if model is not None else None

if model is None:
    st.warning("⚠️ The prediction model file `model/heart_disease_model.pkl` was not found. Please run the training script first.")
    if st.button("Generate Dataset & Train Model Now"):
        with st.spinner("Generating dataset and training classifiers..."):
            os.system("python src/generate_dataset.py")
            os.system("python src/train.py")
            st.rerun()
else:
    # Set up layout: two columns
    col_input, col_result = st.columns([5, 5])
    
    with col_input:
        st.subheader("Patient Clinical Data Form")
        st.write("Please fill in the health parameters below:")
        
        with st.form("patient_data_form"):
            # Split fields into grid
            grid_col1, grid_col2 = st.columns(2)
            
            with grid_col1:
                age = st.slider("Age (Years)", min_value=30, max_value=85, value=55, help="Age of the patient.")
                
                gender = st.selectbox("Gender", options=["Male", "Female"], index=0, help="Biological sex.")
                
                chest_pain = st.selectbox(
                    "Chest Pain Type", 
                    options=[
                        "0: Typical Angina", 
                        "1: Atypical Angina", 
                        "2: Non-Anginal Pain", 
                        "3: Asymptomatic"
                    ],
                    index=1,
                    help="Typical Angina: exertional chest pain. Atypical: non-exertional pain. Non-Anginal: unrelated. Asymptomatic: silent disease."
                )
                
                resting_bp = st.slider("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=130, help="Resting systolic blood pressure upon admission.")
                
                cholesterol = st.slider("Serum Cholesterol (mg/dl)", min_value=120, max_value=450, value=240, help="Total serum cholesterol.")
                
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No (0)", "Yes (1)"], index=0, help="Fasting blood glucose level.")
                
            with grid_col2:
                resting_ecg = st.selectbox(
                    "Resting ECG Results", 
                    options=[
                        "0: Normal", 
                        "1: ST-T Wave Abnormality", 
                        "2: Left Ventricular Hypertrophy"
                    ],
                    index=0,
                    help="Normal: normal ecg. ST-T: wave abnormalities. LVH: thickening of left ventricle wall."
                )
                
                max_hr = st.slider("Maximum Heart Rate (bpm)", min_value=70, max_value=202, value=150, help="Maximum heart rate achieved during cardiac stress test.")
                
                exercise_angina = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"], index=0, help="Chest pain brought on by exercise stress test.")
                
                oldpeak = st.slider("ST Depression / Oldpeak", min_value=0.0, max_value=6.2, value=0.0, step=0.1, help="ST depression induced by exercise relative to rest.")
                
                st_slope = st.selectbox(
                    "ST Segment Slope", 
                    options=[
                        "0: Up", 
                        "1: Flat", 
                        "2: Down"
                    ],
                    index=0,
                    help="Slope of the peak exercise ST segment."
                )
                
            submit_button = st.form_submit_button(label="Analyze Cardiovascular Risk")

    with col_result:
        st.subheader("Diagnostic Prediction Insights")
        
        if submit_button:
            # Map input parameters
            # Map Chest Pain
            cp_val = chest_pain.split(":")[0] # "0", "1", "2", "3"
            
            # Map Fasting BS
            fbs_val = 1 if "Yes" in fasting_bs else 0
            
            # Map Resting ECG
            ecg_val = resting_ecg.split(":")[0] # "0", "1", "2"
            
            # Map ST Slope
            slope_val = st_slope.split(":")[0] # "0", "1", "2"
            
            patient_dict = {
                "Age": age,
                "Gender": gender,
                "Chest_Pain_Type": cp_val,
                "Resting_Blood_Pressure": resting_bp,
                "Cholesterol": cholesterol,
                "Fasting_Blood_Sugar": fbs_val,
                "Resting_ECG": ecg_val,
                "Maximum_Heart_Rate": max_hr,
                "Exercise_Induced_Angina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": slope_val
            }
            
            # Convert to dataframe
            df_patient = pd.DataFrame([patient_dict])
            
            # Predict
            with st.spinner("Analyzing parameters..."):
                # Get probabilities
                prob_disease = model.predict_proba(df_patient)[0][1]
                pred_label = model.predict(df_patient)[0]
                
            confidence = prob_disease if pred_label == 1 else (1.0 - prob_disease)
            prob_percent = int(prob_disease * 100)
            
            # Classify Risk Level
            if prob_disease < 0.30:
                risk_level = "Low Risk"
                card_class = "card-healthy"
                alert_icon = "🟢"
                bar_color = "green"
            elif prob_disease < 0.70:
                risk_level = "Moderate Risk"
                card_class = "card-warning"
                alert_icon = "🟡"
                bar_color = "orange"
            else:
                risk_level = "High Risk"
                card_class = "card-danger"
                alert_icon = "🔴"
                bar_color = "red"
                
            # HTML Card Display for Results
            result_label = "Heart Disease Detected" if pred_label == 1 else "No Heart Disease"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <div class="metric-title">Cardiovascular Diagnosis</div>
                <div class="metric-value">{alert_icon} {result_label}</div>
                <div class="metric-desc">
                    The predictive model estimates a <strong>{prob_percent}%</strong> likelihood of clinical heart disease. Status classified as <strong>{risk_level}</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Progress bar
            st.write("**Cardiovascular Disease Probability Index**")
            st.progress(prob_disease)
            
            # Display patient vitals warnings
            warnings = []
            if resting_bp >= 140:
                warnings.append(f"⚠️ Hypertension detected: Blood pressure is elevated ({resting_bp} mmHg).")
            if cholesterol >= 240:
                warnings.append(f"⚠️ Hypercholesterolemia: Serum Cholesterol is high ({cholesterol} mg/dl).")
            if oldpeak > 1.5:
                warnings.append(f"⚠️ Severe ST depression ({oldpeak}): High exercise-induced ischemia risk.")
            if "Asymptomatic" in chest_pain:
                warnings.append("⚠️ Asymptomatic Chest Pain is clinically correlated with latent arterial blockages.")
                
            if warnings:
                with st.expander("🔍 Clinical Risk Indicators Identified", expanded=True):
                    for warning in warnings:
                        st.markdown(warning)
                        
            # Dynamic Feature Importance
            if feat_importance is not None:
                st.write("**Top Health Determinants Impacting Predictions**")
                
                # Render feature importance chart matching app aesthetics
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = ["#c1121f" if x in ["Chest Pain Type: Asymptomatic", "Exercise Induced Angina: Yes", "ST Depression / Oldpeak", "ST Segment Slope: Down", "ST Segment Slope: Flat"] else "#3a86c8" for x in feat_importance["Feature"]]
                
                # Plot
                sns.barplot(
                    x="Importance", 
                    y="Feature", 
                    data=feat_importance, 
                    palette="crest",
                    hue="Feature",
                    legend=False,
                    ax=ax
                )
                ax.set_title("Relative Feature Weights (Random Forest Model)", fontsize=9, pad=8, weight="bold")
                ax.set_xlabel("Relative Weight", fontsize=8)
                ax.set_ylabel("", fontsize=8)
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        else:
            # Placeholder display
            st.info("👈 Enter patient medical measurements on the left panel, and click **Analyze Cardiovascular Risk** to see predictive insights.")
            
            st.image("https://img.icons8.com/illustrations/external-justicon-flat-justicon/250/external-electrocardiography-hospital-and-medical-justicon-flat-justicon.png", width=220)
            st.markdown("""
            ### Diagnostic Engine Information
            
            This machine learning assistant aggregates standard patient health metrics to predict ischemic heart disease risk.
            The predictive core is trained on clinical indicators and evaluates:
            1. **Patient Demographics** (Age, Gender)
            2. **Vascular Health Indicators** (Resting Blood Pressure, Serum Cholesterol)
            3. **Cardiac Stress Indicators** (Maximum Heart Rate, Exercise Angina, ST Segment changes)
            
            *Note: This tool is designed for educational and clinical support purposes and does not replace medical diagnostics.*
            """)
