import os
import sys
import argparse
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Constants for default healthy/typical values in case some features are omitted
DEFAULT_PATIENT_DATA: Dict[str, Any] = {
    "Age": 50,
    "Gender": "Male",
    "Chest_Pain_Type": "Atypical Angina",
    "Resting_Blood_Pressure": 120,
    "Cholesterol": 200,
    "Fasting_Blood_Sugar": 0,
    "Resting_ECG": "Normal",
    "Maximum_Heart_Rate": 150,
    "Exercise_Induced_Angina": "No",
    "Oldpeak": 0.0,
    "ST_Slope": "Up"
}

def map_categorical_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes categorical inputs from numerical codes or short strings to match dataset values.
    """
    mapped = data.copy()
    
    # 1. Gender mapping
    g = str(mapped.get("Gender", "")).strip().lower()
    if g in ["m", "male", "1"]:
        mapped["Gender"] = "Male"
    elif g in ["f", "female", "0"]:
        mapped["Gender"] = "Female"
        
    # 2. Chest Pain Type mapping (supports 0, 1, 2, 3 or names)
    cp = str(mapped.get("Chest_Pain_Type", "")).strip().lower()
    cp_map = {
        "0": "Typical Angina",
        "typical": "Typical Angina",
        "typical angina": "Typical Angina",
        "1": "Atypical Angina",
        "atypical": "Atypical Angina",
        "atypical angina": "Atypical Angina",
        "2": "Non-Anginal Pain",
        "non-anginal": "Non-Anginal Pain",
        "non-anginal pain": "Non-Anginal Pain",
        "3": "Asymptomatic",
        "asymptomatic": "Asymptomatic"
    }
    if cp in cp_map:
        mapped["Chest_Pain_Type"] = cp_map[cp]
        
    # 3. Fasting Blood Sugar mapping
    fbs = str(mapped.get("Fasting_Blood_Sugar", "")).strip().lower()
    if fbs in ["1", "true", "yes", "y", "high"]:
        mapped["Fasting_Blood_Sugar"] = 1
    elif fbs in ["0", "false", "no", "n", "normal"]:
        mapped["Fasting_Blood_Sugar"] = 0
        
    # 4. Resting ECG mapping
    ecg = str(mapped.get("Resting_ECG", "")).strip().lower()
    ecg_map = {
        "0": "Normal",
        "normal": "Normal",
        "1": "ST-T Wave Abnormality",
        "st": "ST-T Wave Abnormality",
        "st-t wave abnormality": "ST-T Wave Abnormality",
        "2": "Left Ventricular Hypertrophy",
        "lvh": "Left Ventricular Hypertrophy",
        "left ventricular hypertrophy": "Left Ventricular Hypertrophy"
    }
    if ecg in ecg_map:
        mapped["Resting_ECG"] = ecg_map[ecg]
        
    # 5. Exercise Induced Angina mapping
    exang = str(mapped.get("Exercise_Induced_Angina", "")).strip().lower()
    if exang in ["1", "yes", "y", "true"]:
        mapped["Exercise_Induced_Angina"] = "Yes"
    elif exang in ["0", "no", "n", "false"]:
        mapped["Exercise_Induced_Angina"] = "No"
        
    # 6. ST Slope mapping
    slope = str(mapped.get("ST_Slope", "")).strip().lower()
    slope_map = {
        "0": "Up",
        "up": "Up",
        "1": "Flat",
        "flat": "Flat",
        "2": "Down",
        "down": "Down"
    }
    if slope in slope_map:
        mapped["ST_Slope"] = slope_map[slope]
        
    # Convert numerical features
    for num_col in ["Age", "Resting_Blood_Pressure", "Cholesterol", "Maximum_Heart_Rate"]:
        if num_col in mapped and mapped[num_col] is not None:
            try:
                mapped[num_col] = float(mapped[num_col])
            except ValueError:
                pass
                
    if "Oldpeak" in mapped and mapped["Oldpeak"] is not None:
        try:
            mapped["Oldpeak"] = float(mapped["Oldpeak"])
        except ValueError:
            pass
            
    return mapped

def load_prediction_model(model_path: str = "model/heart_disease_model.pkl") -> Any:
    """
    Loads the saved model pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")
    return joblib.load(model_path)

def make_prediction(model: Any, patient_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Performs inference on a patient data dictionary.
    """
    # 1. Fill missing features with defaults
    final_data = DEFAULT_PATIENT_DATA.copy()
    final_data.update(patient_data)
    
    # 2. Map categorical and numerical formats
    final_data = map_categorical_inputs(final_data)
    
    # 3. Create DataFrame (single row)
    df = pd.DataFrame([final_data])
    
    # 4. Predict
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    
    # If prediction is 0, confidence is probability of class 0, else class 1
    confidence = prob if pred == 1 else (1.0 - prob)
    
    return pred, confidence

def get_interactive_input() -> Dict[str, Any]:
    """
    Prompts the user in CLI for patient values.
    """
    print("\n--- Enter Patient Health Parameters ---")
    data = {}
    
    # Ask questions
    try:
        data["Age"] = input("Age (e.g. 45): ")
        data["Gender"] = input("Gender (Male/Female): ")
        data["Chest_Pain_Type"] = input("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic): ")
        data["Resting_Blood_Pressure"] = input("Resting Blood Pressure in mmHg (e.g. 130): ")
        data["Cholesterol"] = input("Serum Cholesterol in mg/dl (e.g. 220): ")
        data["Fasting_Blood_Sugar"] = input("Fasting Blood Sugar > 120 mg/dl? (1: Yes, 0: No): ")
        data["Resting_ECG"] = input("Resting ECG (0: Normal, 1: ST Abnormality, 2: LVH): ")
        data["Maximum_Heart_Rate"] = input("Maximum Heart Rate Achieved (e.g. 150): ")
        data["Exercise_Induced_Angina"] = input("Exercise Induced Angina? (Yes/No): ")
        data["Oldpeak"] = input("ST Depression/Oldpeak (e.g. 1.2): ")
        data["ST_Slope"] = input("ST Slope (0: Up, 1: Flat, 2: Down): ")
        
        # Filter out empty values (they will fallback to defaults)
        data = {k: v for k, v in data.items() if v.strip() != ""}
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")
        sys.exit(0)
        
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Prediction CLI Tool")
    parser.add_argument("--age", type=int, help="Age of patient")
    parser.add_argument("--gender", type=str, help="Gender (Male/Female)")
    parser.add_argument("--cp", type=str, help="Chest Pain Type (0: Typical, 1: Atypical, 2: Non-Anginal, 3: Asymptomatic)")
    parser.add_argument("--bp", type=int, help="Resting Blood Pressure (mmHg)")
    parser.add_argument("--chol", type=int, help="Cholesterol (mg/dl)")
    parser.add_argument("--fbs", type=str, help="Fasting Blood Sugar (1: Yes, 0: No)")
    parser.add_argument("--ecg", type=str, help="Resting ECG (0: Normal, 1: ST, 2: LVH)")
    parser.add_argument("--maxhr", type=int, help="Maximum Heart Rate")
    parser.add_argument("--exang", type=str, help="Exercise Induced Angina (Yes/No)")
    parser.add_argument("--oldpeak", type=float, help="ST Depression (Oldpeak)")
    parser.add_argument("--slope", type=str, help="ST Slope (0: Up, 1: Flat, 2: Down)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive CLI mode")
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_prediction_model()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Check if CLI args are empty
    has_args = any(getattr(args, arg) is not None for arg in vars(args) if arg != "interactive")
    
    if args.interactive or not has_args:
        # Run interactive mode
        patient_data = get_interactive_input()
    else:
        # Run command line argument mode
        patient_data = {}
        if args.age is not None: patient_data["Age"] = args.age
        if args.gender is not None: patient_data["Gender"] = args.gender
        if args.cp is not None: patient_data["Chest_Pain_Type"] = args.cp
        if args.bp is not None: patient_data["Resting_Blood_Pressure"] = args.bp
        if args.chol is not None: patient_data["Cholesterol"] = args.chol
        if args.fbs is not None: patient_data["Fasting_Blood_Sugar"] = args.fbs
        if args.ecg is not None: patient_data["Resting_ECG"] = args.ecg
        if args.maxhr is not None: patient_data["Maximum_Heart_Rate"] = args.maxhr
        if args.exang is not None: patient_data["Exercise_Induced_Angina"] = args.exang
        if args.oldpeak is not None: patient_data["Oldpeak"] = args.oldpeak
        if args.slope is not None: patient_data["ST_Slope"] = args.slope
        
    # Predict
    try:
        pred, confidence = make_prediction(model, patient_data)
        
        # Display output matching requested structure
        result_text = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
        
        print("\n" + "=" * 40)
        print(" prediction result ".upper().center(40, "="))
        print("=" * 40)
        print(f"Prediction: {result_text}")
        print(f"Confidence: {confidence:.0%}")
        print("=" * 40 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
