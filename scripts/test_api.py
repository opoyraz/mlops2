import pandas as pd
import requests

# Load your real test data
df = pd.read_csv("test.csv")  # Use your actual test CSV if separate
X = df.drop(columns=["Resource", "APT Group", "APTGroup"])  # Same preprocessing as training

# Pick a test sample (first row for example)
sample_features = X.iloc[0].tolist()

# Build JSON payload
sample_input = {
    "features": sample_features
}

# Call the API
response = requests.post("http://127.0.0.1:8000/predict", json=sample_input)
print("Response:", response.json())
