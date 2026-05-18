🏠 House Price Prediction

📊 End-to-end Machine Learning project to predict house prices using regression models and deploy them with Streamlit.

✨ Project Highlights

⚡ End-to-end ML pipeline (data → model → deployment)
🧹 Data cleaning & preprocessing
📊 Feature engineering & scaling
🤖 Regression model training
🌐 Interactive web app using Streamlit
💾 Model serialization using Joblib

🎯 Problem Statement

Predict the price of a house based on various features like:

Overall quality of the house
Living area size
Number of bedrooms
Garage area
Year built
Basement area

  **Machine Learning Pipeline**
  
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Scaling (StandardScaler)
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving (.pkl)
      ↓
Streamlit Deployment

⚙️ Tech Stack
Component	                            Tools
Language	                            Python 🐍
Data Handling                        	Pandas, NumPy
Visualization	                        Matplotlib, Seaborn
ML Model                            	Scikit-learn
Deployment                          	Streamlit
Model Saving                        	Joblib

**Run Application**
streamlit run app.py

**  Application Preview****

✨ User-friendly interface where users can:

Enter house features
Click predict
Get instant price prediction

**Sample Input & Output**
🔹 Input
Overall Quality: 7
Living Area: 1500 sq ft
Bedrooms: 3
🔹 Output

💰 Predicted Price: ~$245,000

