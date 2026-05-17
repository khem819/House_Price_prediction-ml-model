## ==========================================
## House Price Prediction Project
## ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

## ==========================================
## Load Dataset
## ==========================================

df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

print("Shape of Train Data :", df_train.shape)
print("Shape of Test Data  :", df_test.shape)

## ==========================================
## Data Integration
## ==========================================

df = pd.concat([df_train, df_test], axis=0)

print("Integrated Data Shape :", df.shape)

## ==========================================
## Feature Types
## ==========================================

int_features = df.select_dtypes(include=["int64"]).columns

float_features = df.select_dtypes(include=["float64"]).columns

cat_features = df.select_dtypes(include=["object"]).columns

print("Integer Features :", len(int_features))
print("Float Features   :", len(float_features))
print("Categorical Features :", len(cat_features))

## ==========================================
## Statistical Information
## ==========================================

print(df.describe())

## ==========================================
## Create Folder For EDA Images
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

print("\nCategorical Missing Features")
print(cat_na_feat)

print("\nInteger Missing Features")
print(int_na_feat)

print("\nFloat Missing Features")
print(float_na_feat)

## ==========================================
## Handle MSZoning Missing Values
## ==========================================

print("\nMSZoning Value Counts")
print(df["MSZoning"].value_counts())

plt.figure(figsize=(10, 6))

df["MSZoning"].value_counts().plot(kind="bar")

plt.xlabel("MSZoning")
plt.ylabel("Count")
plt.title("MSZoning Distribution")

plt.tight_layout()

plt.savefig("EDA_img/mszoning_bar_graph.png")

plt.close()

## Backup Copy
df_mvi = df.copy()

## Fill Missing Values
mszoning_mode = df["MSZoning"].mode()[0]

df_mvi["MSZoning"] = df_mvi["MSZoning"].fillna(mszoning_mode)

## ==========================================
## Handle Numerical Missing Values
## ==========================================

for feature in float_na_feat.index:

    median_value = df_mvi[feature].median()

    df_mvi[feature] = df_mvi[feature].fillna(median_value)

for feature in int_na_feat.index:

    median_value = df_mvi[feature].median()

    df_mvi[feature] = df_mvi[feature].fillna(median_value)

## ==========================================
## Handle Categorical Missing Values
## ==========================================

for feature in cat_na_feat.index:

    mode_value = df_mvi[feature].mode()[0]

    df_mvi[feature] = df_mvi[feature].fillna(mode_value)

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

## Example Plot
boxHistPlot(
    df_mvi["LotFrontage"],
    "EDA_img/boxhist_LotFrontage.png"
)

## ==========================================
## One Hot Encoding
## ==========================================

df_encode = df_mvi.copy()

object_features = df_encode.select_dtypes(
    include=["object"]
).columns.tolist()

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

x_train = df_encode[:len_train].drop("SalePrice", axis=1)

y_train = df_encode[:len_train]["SalePrice"]

x_test = df_encode[len_train:].drop("SalePrice", axis=1)

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

from sklearn.svm import SVR

from sklearn.linear_model import (
    LinearRegression,
    SGDRegressor
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.neural_network import MLPRegressor

from sklearn.isotonic import IsotonicRegression

from xgboost import XGBRegressor

## ==========================================
## Create Models
## ==========================================

models = {

    "a": [
        "LinearRegression",
        LinearRegression()
    ],

    "b": [
        "SVR",
        SVR()
    ],

    "c": [
        "SGDRegressor",
        SGDRegressor(
            max_iter=1000,
            tol=1e-3
        )
    ],

    "d": [
        "KNeighborsRegressor",
        KNeighborsRegressor()
    ],

    "e": [
        "GaussianProcessRegressor",
        GaussianProcessRegressor()
    ],

    "f": [
        "DecisionTreeRegressor",
        DecisionTreeRegressor(
            random_state=42
        )
    ],

    "g": [
        "GradientBoostingRegressor",
        GradientBoostingRegressor(
            random_state=42
        )
    ],

    "h": [
        "RandomForestRegressor",
        RandomForestRegressor(
            random_state=42
        )
    ],

    "i": [
        "XGBRegressor",
        XGBRegressor(
            random_state=42
        )
    ],

    "j": [
        "MLPRegressor",
        MLPRegressor(
            max_iter=1000,
            random_state=42
        )
    ]
}

## ==========================================
## Cross Validation Function
## ==========================================

def test_model(model, x_train, y_train):

    cv = KFold(
        n_splits=7,
        shuffle=True,
        random_state=45
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

for key, value in models.items():

    model_name = value[0]

    model = value[1]

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
## Best Model
## ==========================================

best_model_name = score_df.iloc[0]["Model Name"]

best_model_score = score_df.iloc[0]["R2 Score"]

print("\nBest Model :", best_model_name)

print("Best R2 Score :", best_model_score)