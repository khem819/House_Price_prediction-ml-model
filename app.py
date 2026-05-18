import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 House Price Prediction")

st.write("Enter House Features")

overallqual = st.slider("Overall Quality", 1, 10, 5)
grlivarea = st.number_input("Living Area", value=1500)
garagecars = st.slider("Garage Cars", 0, 5, 2)

if st.button("Predict Price"):

    input_data = np.array([[overallqual, grlivarea, garagecars]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

# Define your base regressor and transformations
regressor = LinearRegression()
transformer_func = np.log
inverse_func = np.exp

# Wrap them in TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=regressor,
    func=transformer_func,
    inverse_func=inverse_func
)
