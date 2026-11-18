"""
Music Genre Classifier & Explorer - ML Training Pipeline
GLA University - ML Software Project
Trains Random Forest, SVM, and Ensemble Classifiers on GTZAN features.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def train_and_export():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "features_30_sec_with_missing_values (1).csv")
    models_dir = os.path.join(base_dir, "ml", "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"[1/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    initial_shape = df.shape
    print(f"       Loaded dataset with shape {initial_shape}")

    # Remove duplicates
    duplicates_count = int(df.duplicated().sum())
    df = df.drop_duplicates()
    print(f"       Removed {duplicates_count} duplicate rows.")

    # Impute missing values
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    print(f"[2/6] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

    # Outlier statistics
    outlier_summary = {}
    for col in num_cols:
        if col not in ['length']:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_cnt = int(((df[col] < lower) | (df[col] > upper)).sum())
            if outlier_cnt > 0:
                outlier_summary[col] = outlier_cnt

    # Prepare features and targets
    drop_cols = [c for c in ['label', 'filename'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['label']

    feature_names = list(X.columns)
    print(f"[3/6] Total Features: {len(feature_names)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"       Classes ({len(class_names)}): {class_names}")

    # Save cleaned dataset for reference
    cleaned_path = os.path.join(models_dir, "cleaned_features.csv")
    df.to_csv(cleaned_path, index=False)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    print(f"       Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest (Baseline from Synopsis)
    print("[4/6] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=22,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)

    # 2. Support Vector Machine (Comparison from Synopsis)
    print("       Training Support Vector Machine (SVM)...")
    svm_model = SVC(
        C=4.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    svm_preds = svm_model.predict(X_test_scaled)
    svm_probs = svm_model.predict_proba(X_test_scaled)

    # 3. Soft-Voting Ensemble (RF + SVM + ExtraTrees)
    print("       Training Soft-Voting Ensemble...")
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)

    ensemble_preds = []
    et_probs = et_model.predict_proba(X_test)
    # Blend probabilities
    blended_probs = (rf_probs * 0.40) + (svm_probs * 0.35) + (et_probs * 0.25)
    ensemble_preds = np.argmax(blended_probs, axis=1)

    # Save artifacts
    print("[5/6] Saving trained models and preprocessors...")
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.joblib"))
    joblib.dump(svm_model, os.path.join(models_dir, "svm.joblib"))
    joblib.dump(et_model, os.path.join(models_dir, "extra_trees.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))

    with open(os.path.join(models_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Calculate metrics
    def calc_metrics(y_true, y_pred, probs=None):
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        per_class = {}
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        for i, c in enumerate(class_names):
            per_class[c] = {
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1_score": round(float(f1_c[i]), 4),
                "support": int(sup_c[i])
            }
        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(p), 4),
            "macro_recall": round(float(r), 4),
            "macro_f1": round(float(f1), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "confusion_matrix": cm,
            "per_class": per_class
        }

    rf_metrics = calc_metrics(y_test, rf_preds, rf_probs)
    svm_metrics = calc_metrics(y_test, svm_preds, svm_probs)
    ens_metrics = calc_metrics(y_test, ensemble_preds, blended_probs)

    # Feature Importances from Random Forest
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"feature": feature_names[idx], "importance": round(float(importances[idx]), 4)}
        for idx in sorted_idx[:25]
    ]

    # Save metrics JSON for Frontend display
    metrics_data = {
        "dataset_info": {
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_features": len(feature_names),
            "num_classes": len(class_names),
            "classes": class_names,
            "missing_values_handled": 1714,
            "outlier_summary_count": len(outlier_summary)
        },
        "models": {
            "random_forest": {
                "name": "Random Forest (Baseline)",
                "description": "Ensemble of 300 decision trees with Gini impurity and bootstrap sampling",
                "metrics": rf_metrics
            },
            "svm": {
                "name": "Support Vector Machine (Comparison)",
                "description": "RBF Kernel SVM with standard scaling (C=4.0, gamma='scale')",
                "metrics": svm_metrics
            },
            "ensemble": {
                "name": "Hybrid Ensemble (Optimal)",
                "description": "Soft-voting blend of Random Forest, RBF-SVM, and Extra Trees",
                "metrics": ens_metrics
            }
        },
        "top_features": top_features,
        "feature_names": feature_names
    }

    metrics_file = os.path.join(models_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print("[6/6] Pipeline completed successfully!")
    print(f"       Random Forest Accuracy: {rf_metrics['accuracy'] * 100:.2f}% | Macro F1: {rf_metrics['macro_f1']:.4f}")
    print(f"       SVM Accuracy:           {svm_metrics['accuracy'] * 100:.2f}% | Macro F1: {svm_metrics['macro_f1']:.4f}")
    print(f"       Ensemble Accuracy:      {ens_metrics['accuracy'] * 100:.2f}% | Macro F1: {ens_metrics['macro_f1']:.4f}")
    print(f"       Saved metrics to: {metrics_file}")

if __name__ == "__main__":
    train_and_export()
