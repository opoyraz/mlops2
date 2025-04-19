import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('/mlops/virusTotal/mlops2/dataset.csv')
X = df.drop(columns=["Resource", "APT Group", "APTGroup"])
y = df["APTGroup"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

mlflow.set_experiment("APT_Group_Classification")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

os.makedirs("scripts", exist_ok=True)
joblib.dump(model, "scripts/apt_group_rf_model.joblib")
