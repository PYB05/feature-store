"""
============================================================
ML Feature Store -- Model Training Script
============================================================
Trains an XGBoost classifier on the Credit Card Fraud
Detection dataset and saves the model + scaler to disk.

Usage:
    python train.py

What this does:
    1. Loads the raw CSV (284,807 transactions)
    2. Separates features (V1-V28, Time, Amount) from label (Class)
    3. Scales the 'Time' and 'Amount' columns (V1-V28 are already PCA'd)
    4. Splits 80/20 with stratification to preserve class ratio
    5. Trains XGBoost with scale_pos_weight to handle the extreme
       imbalance (only 0.17% are fraud)
    6. Prints a full classification report
    7. Saves the trained model and scaler as .pkl files
============================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib


def load_data(filepath: str) -> pd.DataFrame:
    """Load the credit card transaction dataset."""
    print(f"[1/6] Loading dataset from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at {filepath}")
        print("Please download creditcard.csv from Kaggle:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  and place it in: {filepath}")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"       Loaded {len(df):,} transactions")
    print(f"       Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"       Legitimate:  {(df['Class'] == 0).sum():,} ({(1 - df['Class'].mean())*100:.3f}%)")
    return df


def preprocess(df: pd.DataFrame):
    """
    Separate features and labels, then scale Time and Amount.
    
    V1-V28 are already PCA-transformed and scaled, but Time and Amount
    are raw values that need standardization for the model to work well.
    """
    print("[2/6] Preprocessing features...")
    
    # Separate features (X) from label (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Time and Amount -- these are the only raw features
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    print(f"       Features: {X.shape[1]} columns")
    print(f"       Scaled: Time, Amount")
    return X, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test with stratification.
    
    Stratification ensures both sets maintain the same fraud ratio (0.17%).
    Without it, the test set might have too few fraud cases to evaluate.
    """
    print(f"[3/6] Splitting data (train={1-test_size:.0%}, test={test_size:.0%}, stratified)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserves class distribution in both sets
    )
    
    print(f"       Train: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    print(f"       Test:  {len(X_test):,} samples ({y_test.sum():,} fraud)")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train an XGBoost classifier with imbalance handling.
    
    XGBoost uses gradient boosting: it trains trees sequentially,
    where each new tree specifically focuses on correcting the
    mistakes made by previous trees. This is far more effective
    than Random Forest for catching rare fraud patterns.

    scale_pos_weight controls how aggressively the model prioritizes
    catching fraud. Too high and it over-predicts fraud (low precision);
    too low and it misses fraud (low recall). The value 10 was selected
    via 5-fold cross-validation to maximize F1 score.
    """
    print("[4/6] Training XGBoost (n_estimators=300, scale_pos_weight=10)...")
    
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=10,           # Tuned via cross-validation
        subsample=0.8,                 # Row sampling for regularization
        colsample_bytree=0.8,          # Column sampling for regularization
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,                     # Use all CPU cores
        use_label_encoder=False,
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"       Training completed in {train_time:.2f} seconds")
    return model


def evaluate_model(model, X_test, y_test):
    """Print comprehensive evaluation metrics."""
    print("[5/6] Evaluating model performance...")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Classification Report — shows precision, recall, F1 per class
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion Matrix — shows true/false positives/negatives
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"   True Negatives:  {cm[0][0]:,}  (correctly identified as legit)")
    print(f"   False Positives: {cm[0][1]:,}  (legit flagged as fraud)")
    print(f"   False Negatives: {cm[1][0]:,}  (fraud missed!)")
    print(f"   True Positives:  {cm[1][1]:,}  (correctly caught fraud)")
    
    # ROC-AUC Score — overall model quality metric
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print("=" * 60)
    
    return y_pred, y_prob


def save_artifacts(model, scaler, model_dir="models"):
    """Save trained model and scaler to disk."""
    print(f"[6/6] Saving artifacts to {model_dir}/...")
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Print file sizes for verification
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    scaler_size = os.path.getsize(scaler_path) / 1024
    
    print(f"       Model saved:  {model_path} ({model_size:.1f} MB)")
    print(f"       Scaler saved: {scaler_path} ({scaler_size:.1f} KB)")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("ML Feature Store -- Fraud Detection Model Training")
    print("=" * 60 + "\n")
    
    # Step 1: Load data
    df = load_data("data/creditcard.csv")
    
    # Step 2: Preprocess
    X, y, scaler = preprocess(df)
    
    # Step 3: Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Step 6: Save
    save_artifacts(model, scaler)
    
    print("\nTraining pipeline complete!")
    print("   Next step: docker compose up --build\n")


if __name__ == "__main__":
    main()
