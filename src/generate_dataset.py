import os
import numpy as np
import pandas as pd
from typing import Optional

def generate_heart_disease_dataset(
    n_samples: int = 1000, 
    missing_rate: float = 0.03, 
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic heart disease dataset with medical correlations
    and randomly injected missing values (~3%).
    """
    np.random.seed(random_seed)
    
    # 1. Generate base features with realistic distributions
    age = np.random.normal(loc=55, scale=10, size=n_samples).astype(int)
    age = np.clip(age, 30, 85)
    
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.6, 0.4])
    
    chest_pain = np.random.choice(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], 
        size=n_samples, 
        p=[0.2, 0.25, 0.2, 0.35]
    )
    
    resting_bp = np.random.normal(loc=130, scale=18, size=n_samples).astype(int)
    resting_bp = np.clip(resting_bp, 90, 200)
    
    cholesterol = np.random.normal(loc=240, scale=45, size=n_samples).astype(int)
    cholesterol = np.clip(cholesterol, 120, 450)
    
    fasting_bs = np.random.choice([1, 0], size=n_samples, p=[0.15, 0.85])
    
    resting_ecg = np.random.choice(
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], 
        size=n_samples, 
        p=[0.55, 0.15, 0.30]
    )
    
    # Max HR is negatively correlated with age: standard estimate is 220 - age
    max_hr_base = 220 - age
    max_hr = (max_hr_base - np.random.normal(loc=20, scale=15, size=n_samples)).astype(int)
    max_hr = np.clip(max_hr, 70, 202)
    
    exercise_angina = np.random.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    
    oldpeak = np.abs(np.random.exponential(scale=1.0, size=n_samples))
    oldpeak = np.clip(oldpeak, 0.0, 6.2).round(1)
    
    st_slope = np.random.choice(["Up", "Flat", "Down"], size=n_samples, p=[0.45, 0.45, 0.10])
    
    # 2. Calculate probabilities for Heart_Disease using a logistic function to ensure realistic correlations
    # Initialize log-odds Z
    z = -3.5
    
    # Age effect (increases risk with age)
    z += 0.04 * (age - 45)
    
    # Gender effect (males have higher risk)
    z += 0.6 * (gender == "Male")
    
    # Chest pain type (asymptomatic chest pain is highly correlated with blockages/disease in studies)
    z += 1.2 * (chest_pain == "Asymptomatic")
    z += 0.3 * (chest_pain == "Typical Angina")
    
    # resting blood pressure (higher BP increases risk)
    z += 0.015 * (resting_bp - 120)
    
    # Cholesterol (higher cholesterol increases risk)
    z += 0.005 * (cholesterol - 200)
    
    # Fasting blood sugar (diabetic state increases risk)
    z += 0.6 * fasting_bs
    
    # ECG abnormalities increase risk
    z += 0.4 * (resting_ecg != "Normal")
    
    # Maximum heart rate (lower max HR under stress is a sign of disease)
    z += -0.02 * (max_hr - 150)
    
    # Exercise induced angina (very strong risk indicator)
    z += 1.0 * (exercise_angina == "Yes")
    
    # ST depression (higher oldpeak increases risk)
    z += 0.8 * oldpeak
    
    # ST slope (Flat/Down indicates higher risk)
    z += 1.2 * (st_slope == "Flat")
    z += 1.8 * (st_slope == "Down")
    
    # Logistic function
    prob = 1 / (1 + np.exp(-z))
    
    # Add minor noise to make it realistic and not perfectly separable
    noise = np.random.normal(loc=0.0, scale=0.1, size=n_samples)
    prob_noisy = np.clip(prob + noise, 0.0, 1.0)
    
    # Generate binary label
    heart_disease = (prob_noisy > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Chest_Pain_Type": chest_pain,
        "Resting_Blood_Pressure": resting_bp,
        "Cholesterol": cholesterol,
        "Fasting_Blood_Sugar": fasting_bs,
        "Resting_ECG": resting_ecg,
        "Maximum_Heart_Rate": max_hr,
        "Exercise_Induced_Angina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
        "Heart_Disease": heart_disease
    })
    
    # 3. Inject ~3% missing values randomly into features
    # Columns to inject missing values (excluding the target column)
    feature_cols = [c for c in df.columns if c != "Heart_Disease"]
    
    for col in feature_cols:
        mask = np.random.rand(n_samples) < missing_rate
        # For numerical columns, we use np.nan, for categorical we use None/nan which pandas will treat as missing
        if df[col].dtype == object or df[col].dtype == 'category':
            df.loc[mask, col] = None
        else:
            df.loc[mask, col] = np.nan
            
    return df

if __name__ == "__main__":
    print("Generating synthetic heart disease dataset...")
    df = generate_heart_disease_dataset(n_samples=1020, missing_rate=0.03, random_seed=42)
    
    # Create dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join("dataset", "heart_disease.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully saved to {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Target distribution (Heart Disease = 1): {df['Heart_Disease'].mean():.2%}")
    print("\nMissing values count per column:")
    print(df.isnull().sum())
