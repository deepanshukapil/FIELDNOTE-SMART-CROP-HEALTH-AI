import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    p = argparse.ArgumentParser(description="Train crop recommendation/risk ML model from labeled CSV")
    p.add_argument("--csv", default="data/crop_recommendation.csv")
    p.add_argument("--out", default="models/crop_risk.joblib")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    target = next((c for c in ["label", "crop", "target"] if c in df.columns), None)
    missing = [c for c in FEATURES if c not in df.columns]
    if target is None or missing:
        raise SystemExit(f"CSV needs features {FEATURES} and one target column named label/crop/target; missing={missing}")
    df = df[FEATURES + [target]].dropna()
    if df[target].nunique() < 2:
        raise SystemExit("Training CSV must contain at least two crop classes")
    X, y = df[FEATURES], df[target].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Holdout accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Saved trained model: {out}")

if __name__ == "__main__": main()
