import pandas as pd
import numpy as np
from pandas import DataFrame

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


raw_df = pd.read_csv('titanic.csv')
print(type(raw_df))

raw_df.info()
print(raw_df.isnull().sum())

# Data cleaning and feature engineering
def data_preprocessing(df: DataFrame) -> DataFrame:
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    df['Embarked'].fillna("S", inplace=True)
    df.drop(columns=['Embarked'], inplace=True)

    fill_missing_ages(df)

    # convert genders
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Add columns
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = np.where(df['FamilySize'] == 0, 1, 0)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    return df

# Fill in missing ages
def fill_missing_ages(df: DataFrame):
    age_fill_map = {}
    for pclass in df['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()

    df['Age'] = df.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) else row['Age'], axis=1)


data = data_preprocessing(raw_df)

# Create features / targets variables (Flash Cards)
x = data.drop(columns=['Survived'])
y = data['Survived']

y = y[x.notna().all(axis=1)]
x.dropna(inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# ML Preprocessing
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# HyperParameter Tuning  - KNN Model
def tune_knn_model(X_train, Y_train) -> KNeighborsClassifier:
    knn_params = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid=knn_params, cv=5, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

best_model = tune_knn_model(x_train_scaled, y_train)


# Predictions and Evaluate
def evaluate_model(model: KNeighborsClassifier, X_test, Y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    matrix = confusion_matrix(Y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test_scaled, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Confusion Matrix:\n{matrix}")




