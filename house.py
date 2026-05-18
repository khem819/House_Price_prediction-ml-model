## ==========================================
## House Price Prediction Project - Final Code
## ==========================================

## ==========================================
## Import Libraries
## ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

## ==========================================
## Load Dataset
## ==========================================

df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

print("Shape of Train Data :", df_train.shape)
print("Shape of Test Data  :", df_test.shape)

## ==========================================
## Combine Train and Test Data
## ==========================================

df = pd.concat([df_train, df_test], axis=0)

print("Integrated Data Shape :", df.shape)

## ==========================================
## Feature Types
## ==========================================

int_features = df.select_dtypes(include=["int64"]).columns
float_features = df.select_dtypes(include=["float64"]).columns
cat_features = df.select_dtypes(include=["object", "string"]).columns

print("Integer Features :", len(int_features))
print("Float Features   :", len(float_features))
print("Categorical Features :", len(cat_features))

## ==========================================
## Create Folder for EDA Images
## ==========================================

os.makedirs("EDA_img", exist_ok=True)

## ==========================================
## Missing Value Heatmap
## ==========================================

plt.figure(figsize=(16, 8))

sns.heatmap(df.isnull())

plt.title("Missing Value Heatmap")

plt.savefig("EDA_img/heatmap_DF_of_null_value.png")

plt.close()

## ==========================================
## Missing Value Percentage
## ==========================================

null_percent = (df.isnull().sum() / df.shape[0]) * 100

missing_value_feat = null_percent[null_percent > 0]

print("\nTotal Missing Features :", len(missing_value_feat))

## ==========================================
## Missing Features By Type
## ==========================================

cat_na_feat = missing_value_feat[
    missing_value_feat.index.isin(cat_features)
]

int_na_feat = missing_value_feat[
    missing_value_feat.index.isin(int_features)
]

float_na_feat = missing_value_feat[
    missing_value_feat.index.isin(float_features)
]

## ==========================================
## Copy Dataset
## ==========================================

df_mvi = df.copy()

## ==========================================
## Fill Numerical Missing Values
## ==========================================

for feature in float_na_feat.index:

    median_value = df_mvi[feature].median()

    df_mvi[feature] = df_mvi[feature].fillna(
        median_value
    )

for feature in int_na_feat.index:

    median_value = df_mvi[feature].median()

    df_mvi[feature] = df_mvi[feature].fillna(
        median_value
    )

## ==========================================
## Fill Categorical Missing Values
## ==========================================

for feature in cat_na_feat.index:

    mode_value = df_mvi[feature].mode()[0]

    df_mvi[feature] = df_mvi[feature].fillna(
        mode_value
    )

## ==========================================
## Boxplot + Histogram Function
## ==========================================

def boxHistPlot(data, filename, figsize=(16, 5)):

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)

    sns.boxplot(x=data)

    plt.subplot(1, 2, 2)

    sns.histplot(data, kde=True)

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()

## ==========================================
## Example Plot
## ==========================================

boxHistPlot(
    df_mvi["LotFrontage"],
    "EDA_img/boxhist_LotFrontage.png"
)

## ==========================================
## Log Transformation
## ==========================================

df_mvi["SalePrice"] = np.log1p(
    df_mvi["SalePrice"]
)

## ==========================================
## One Hot Encoding
## ==========================================

df_encode = df_mvi.copy()

object_features = df_encode.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nTotal Object Features :", len(object_features))

print("\nShape Before Encoding :", df_encode.shape)

df_encode = pd.get_dummies(
    df_encode,
    columns=object_features,
    drop_first=True
)

print("Shape After Encoding :", df_encode.shape)

## ==========================================
## Split Train and Test Data
## ==========================================

len_train = df_train.shape[0]

x_train = df_encode[:len_train].drop(
    "SalePrice",
    axis=1
)

y_train = df_encode[:len_train]["SalePrice"]

x_test = df_encode[len_train:].drop(
    "SalePrice",
    axis=1
)

## Save Feature Names
x_train_columns = x_train.columns

print("\nX Train Shape :", x_train.shape)
print("Y Train Shape :", y_train.shape)
print("X Test Shape  :", x_test.shape)

## ==========================================
## Feature Scaling
## ==========================================

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

print("\nFeature Scaling Completed")

## ==========================================
## Import ML Models
## ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

## ==========================================
## Create Models
## ==========================================

models = {

    "LinearRegression": LinearRegression(),

    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "GradientBoostingRegressor": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBRegressor": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
}

## ==========================================
## Cross Validation Function
## ==========================================

def test_model(model, x_train, y_train):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    r2 = make_scorer(r2_score)

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=r2
    )

    return scores.mean()

## ==========================================
## Train All Models
## ==========================================

models_score = []

for model_name, model in models.items():

    print("\n" + "=" * 60)

    print("Training Model :", model_name)

    score = test_model(
        model,
        x_train,
        y_train
    )

    print("R2 Score :", score)

    models_score.append(
        [model_name, score]
    )

## ==========================================
## Final Result DataFrame
## ==========================================

score_df = pd.DataFrame(
    models_score,
    columns=[
        "Model Name",
        "R2 Score"
    ]
)

score_df = score_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("Final Model Performance")
print("=" * 60)

print(score_df)

## ==========================================
## Select Best Model
## ==========================================

best_model_name = score_df.iloc[0]["Model Name"]

print("\nBest Model :", best_model_name)

## ==========================================
## Initialize Best Model
## ==========================================

if best_model_name == "LinearRegression":

    best_model = LinearRegression()

elif best_model_name == "RandomForestRegressor":

    best_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

elif best_model_name == "GradientBoostingRegressor":

    best_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

else:

    best_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

## ==========================================
## Train Best Model
## ==========================================

best_model.fit(x_train, y_train)

print("\nBest Model Trained Successfully")

## ==========================================
## Training Prediction
## ==========================================

train_pred = best_model.predict(x_train)

## Reverse Log Transform
train_pred_actual = np.expm1(train_pred)

y_train_actual = np.expm1(y_train)

## ==========================================
## Model Evaluation
## ==========================================

mae = mean_absolute_error(
    y_train_actual,
    train_pred_actual
)

rmse = np.sqrt(
    mean_squared_error(
        y_train_actual,
        train_pred_actual
    )
)

r2 = r2_score(
    y_train_actual,
    train_pred_actual
)

print("\n")
print("=" * 60)
print("Model Evaluation")
print("=" * 60)

print("MAE :", mae)
print("RMSE :", rmse)
print("R2 Score :", r2)

## ==========================================
## Feature Importance
## ==========================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": x_train_columns,

        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features")

    print(importance.head(20))

## ==========================================
## Predict Test Data
## ==========================================

y_pred = best_model.predict(x_test)

## Reverse Log Transform
y_pred = np.expm1(y_pred)

print("\nPrediction Completed")

## ==========================================
## Create Submission File
## ==========================================

submission = pd.DataFrame({

    "Id": df_test["Id"],

    "SalePrice": y_pred
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission File Created")

## ==========================================
## Save Model and Scaler
## ==========================================

joblib.dump(best_model,"house_price_model.pkl")
joblib.dump(sc,"scaler.pkl")
# Save feature/column names used by the model
joblib.dump(list(x_train_columns), "model_columns.pkl")

print("\nModel Saved Successfully")

print("house_price_model.pkl")
print("scaler.pkl")

## ==========================================
## Final Message
## ==========================================

print("\n")
print("=" * 60)
print("Project Completed Successfully")
print("=" * 60)