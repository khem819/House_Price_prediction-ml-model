import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

# Load dataset
df = pd.read_csv("./data/train.csv")

# Use only 3 features
features = ["OverallQual", "GrLivArea", "GarageCars"]

X = df[features]
y = df["SalePrice"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
base_model = LinearRegression()

model = TransformedTargetRegressor(
    regressor=base_model,
    func=np.log,
    inverse_func=np.exp
)

# Train
model.fit(X_scaled, y)

# Save files
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "model_columns.pkl")

print("Model Saved Successfully")