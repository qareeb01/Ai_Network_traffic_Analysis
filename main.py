# ============================================================
# AI-Driven Network Traffic Analysis for Cyber Threat Detection
# Dataset: CICIDS2017 (DDoS Friday Afternoon)
# Model:   Random Forest Classifier
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# ─── CONFIG ─────────────────────────────────────────────────
MODEL_PATH   = "model.pkl"
LABEL_COL    = "Label"        # after stripping spaces
TEST_SIZE    = 0.2
RANDOM_STATE = 42
# ────────────────────────────────────────────────────────────


# ----------------------------------------------------------
# Step 1: Load the dataset
# ----------------------------------------------------------
def load_data(filepath):
    """Read CSV file into a pandas DataFrame."""
    print(f"[1] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"    Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


# ----------------------------------------------------------
# Step 2: Clean the dataset
# ----------------------------------------------------------
def clean_data(df):
    """
    - Strip leading/trailing spaces from column names
    - Replace infinite values with NaN, then drop all NaN rows
    """
    print("[2] Cleaning dataset...")

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Replace inf / -inf with NaN so we can drop them easily
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)

    print(f"    Removed {before - after:,} rows with missing/infinite values.")
    print(f"    Remaining rows: {after:,}")
    return df


# ----------------------------------------------------------
# Step 3: Prepare features (X) and target (y)
# ----------------------------------------------------------
def prepare_features(df, label_col):
    """
    - Separate features (X) from the label column (y)
    - Encode the label column from text to numbers
    """
    print("[3] Preparing features and encoding labels...")

    # Separate features and labels
    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    # Encode text labels → integers  (e.g. 'BENIGN' → 0, 'DDoS' → 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print(f"    Feature columns : {X.shape[1]}")
    print(f"    Label classes   : {list(encoder.classes_)}")
    return X, y, encoder


# ----------------------------------------------------------
# Step 4: Split data into train and test sets
# ----------------------------------------------------------
def split_data(X, y):
    """Split dataset: 80% training, 20% testing."""
    print("[4] Splitting data into train/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y           # keep class balance in both splits
    )

    print(f"    Training samples : {len(X_train):,}")
    print(f"    Testing  samples : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------
# Step 5: Train the Random Forest model
# ----------------------------------------------------------
def train_model(X_train, y_train):
    """Train a Random Forest classifier on the training data."""
    print("[5] Training RandomForestClassifier...")

    model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    print("    Training complete.")
    return model


# ----------------------------------------------------------
# Step 6: Evaluate the model
# ----------------------------------------------------------
def evaluate_model(model, X_test, y_test, encoder):
    """Print accuracy and a full classification report."""
    print("[6] Evaluating model on test set...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n    Accuracy: {acc * 100:.2f}%\n")

    report = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_
    )
    print("    Classification Report:")
    print(report)


# ----------------------------------------------------------
# Step 7: Save the trained model
# ----------------------------------------------------------
def save_model(model, path):
    """Serialize the trained model to a .pkl file with joblib."""
    print(f"[7] Saving model to: {path}")
    joblib.dump(model, path)
    print("    Model saved successfully.")


# ----------------------------------------------------------
# Main – runs all steps in order
# ----------------------------------------------------------
def main():
    print("=" * 55)
    print(" Network Traffic Threat Detection — CICIDS2017")
    print("=" * 55)

    df             = load_data(DATASET_PATH)
    df             = clean_data(df)
    X, y, encoder  = prepare_features(df, LABEL_COL)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model          = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, encoder)
    save_model(model, MODEL_PATH)

    print("\nDone! ✓")


if __name__ == "__main__":
    main()