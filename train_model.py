import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

# 2. Split features and target
X = df.drop("label", axis=1)
y = df["label"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Save trained model
joblib.dump(model, "model/crop_model.pkl")

print("✅ Model trained and saved successfully")
