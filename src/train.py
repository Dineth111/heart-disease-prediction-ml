import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)

# Output directory for images
IMAGE_DIR = os.path.join("notebooks", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define feature categories
NUMERICAL_FEATURES: List[str] = [
    "Age", 
    "Resting_Blood_Pressure", 
    "Cholesterol", 
    "Maximum_Heart_Rate", 
    "Oldpeak"
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender", 
    "Chest_Pain_Type", 
    "Fasting_Blood_Sugar", 
    "Resting_ECG", 
    "Exercise_Induced_Angina", 
    "ST_Slope"
]

# ==========================================
# DATA PREPROCESSING SECTION
# ==========================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset successfully loaded from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns a ColumnTransformer preprocessing pipeline.
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor

def preprocess_data(
    df: pd.DataFrame, 
    target_col: str = "Heart_Disease",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Splits the dataset and returns raw training/testing sets alongside the preprocessor.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    missing_counts = X.isnull().sum()
    print("Missing values in features before imputation:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df):.1%})")
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
    preprocessor = get_preprocessor()
    
    return X_train, X_test, y_train, y_test, preprocessor

# ==========================================
# EVALUATION & METRICS SECTION
# ==========================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculates classification metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    }

def print_evaluation_summary(model_name: str, metrics: Dict[str, float], report: str, cm: np.ndarray) -> None:
    """
    Prints a detailed evaluation summary to the console.
    """
    print("=" * 60)
    print(f" MODEL PERFORMANCE: {model_name} ".center(60, "="))
    print("=" * 60)
    for metric_name, val in metrics.items():
        print(f"{metric_name:<15}: {val:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60 + "\n")

def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_dir: str) -> None:
    """
    Plots and saves a confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticklabels=["No Disease (0)", "Disease (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_dir: str) -> None:
    """
    Plots and saves the ROC curve.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"Receiver Operating Characteristic - {model_name}", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    st_style = ":"
    plt.grid(True, linestyle=st_style, alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def clean_feature_names(feature_names: List[str]) -> List[str]:
    """
    Cleans scikit-learn get_feature_names_out output prefixes.
    """
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

def plot_feature_importance(
    model: Any, 
    preprocessor: Any, 
    model_name: str, 
    save_dir: str
) -> None:
    """
    Extracts and plots feature importances or coefficients.
    """
    importances = None
    title = "Feature Importance"
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importance - {model_name}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"Feature Importance (Absolute Coefficients) - {model_name}"
    
    if importances is None:
        print(f"Skipping feature importance plot for {model_name}: No feature importances.")
        return
        
    raw_feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = clean_feature_names(raw_feature_names)
    
    feat_df = pd.DataFrame({
        "Feature": cleaned_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    feat_df = feat_df.head(15)
    
    plt.figure(figsize=(9, 6))
    sns.barplot(
        x="Importance", 
        y="Feature", 
        data=feat_df, 
        palette="viridis",
        hue="Feature",
        legend=False
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Relative Importance / Absolute Weight", fontsize=12)
    plt.ylabel("Health Feature", fontsize=12)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(comparison_df: pd.DataFrame, save_dir: str) -> None:
    """
    Plots a bar chart comparing performance metrics across different models.
    """
    df_melted = comparison_df.reset_index().melt(
        id_vars="index", 
        var_name="Metric", 
        value_name="Score"
    ).rename(columns={"index": "Model"})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Model", 
        y="Score", 
        hue="Metric", 
        data=df_melted, 
        palette="Set2"
    )
    plt.title("Classifier Performance Comparison", fontsize=14, pad=15)
    plt.ylim([0.0, 1.1])
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Machine Learning Model", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# EDA VISUALIZATION SECTION
# ==========================================

def generate_eda_plots(df: pd.DataFrame) -> None:
    """
    Generates and saves EDA visualizations.
    """
    print("Generating exploratory data analysis (EDA) plots...")
    
    # Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"].dropna(), kde=True, color="skyblue", bins=20)
    plt.title("Distribution of Patient Age", fontsize=14, pad=15)
    plt.xlabel("Age (Years)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "age_distribution.png"), dpi=300)
    plt.close()
    
    # Cholesterol distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Cholesterol"].dropna(), kde=True, color="salmon", bins=20)
    plt.title("Distribution of Serum Cholesterol", fontsize=14, pad=15)
    plt.xlabel("Cholesterol (mg/dl)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "cholesterol_distribution.png"), dpi=300)
    plt.close()
    
    # Correlation Heatmap
    numeric_cols = ["Age", "Resting_Blood_Pressure", "Cholesterol", "Maximum_Heart_Rate", "Oldpeak", "Heart_Disease"]
    corr_matrix = df[numeric_cols].dropna().corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 11}
    )
    plt.title("Correlation Matrix of Numerical Features & Target", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"EDA plots successfully saved to {IMAGE_DIR}")

# ==========================================
# MAIN TRAINING RUN SECTION
# ==========================================

def train_and_compare_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series,
    preprocessor: Any
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Fits preprocessor, trains and compares classifiers, and returns best pipeline.
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        "Support Vector Machine": SVC(probability=True, random_state=42, kernel="rbf")
    }
    
    results = {}
    fitted_classifiers = {}
    
    for name, clf in models.items():
        print(f"Training model: {name}...")
        clf.fit(X_train_processed, y_train)
        fitted_classifiers[name] = clf
        
        y_pred = clf.predict(X_test_processed)
        y_prob = clf.predict_proba(X_test_processed)[:, 1]
        
        metrics = calculate_metrics(y_test.values, y_pred, y_prob)
        results[name] = metrics
        
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print_evaluation_summary(name, metrics, report, cm)
        
    comparison_df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print(" MODEL PERFORMANCE SUMMARY ".center(60, "="))
    print("=" * 60)
    print(comparison_df.round(4))
    print("=" * 60 + "\n")
    
    plot_model_comparison(comparison_df, IMAGE_DIR)
    
    best_model_name = comparison_df["F1_Score"].idxmax()
    best_clf = fitted_classifiers[best_model_name]
    print(f"Selected Best Model: {best_model_name} (F1 Score: {comparison_df.loc[best_model_name, 'F1_Score']:.4f})")
    
    best_y_pred = best_clf.predict(X_test_processed)
    best_y_prob = best_clf.predict_proba(X_test_processed)[:, 1]
    best_cm = confusion_matrix(y_test, best_y_pred)
    
    plot_confusion_matrix(best_cm, best_model_name, IMAGE_DIR)
    plot_roc_curve(y_test.values, best_y_prob, best_model_name, IMAGE_DIR)
    
    if hasattr(best_clf, "feature_importances_") or hasattr(best_clf, "coef_"):
        plot_feature_importance(best_clf, preprocessor, best_model_name, IMAGE_DIR)
    else:
        print(f"Model {best_model_name} does not support direct feature importances. Generating Random Forest feature importances as a fallback.")
        plot_feature_importance(fitted_classifiers["Random Forest"], preprocessor, "Random Forest", IMAGE_DIR)
    
    best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_clf)
    ])
    
    return best_pipeline, comparison_df

if __name__ == "__main__":
    dataset_path = os.path.join("dataset", "heart_disease.csv")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run `python src/generate_dataset.py` first.")
        sys.exit(1)
        
    df = load_data(dataset_path)
    generate_eda_plots(df)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    best_pipeline, comparison_df = train_and_compare_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    os.makedirs("model", exist_ok=True)
    model_save_path = os.path.join("model", "heart_disease_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Best model pipeline successfully saved to {model_save_path}")
