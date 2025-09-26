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

    df['Embarked'].fillna('S', inplace=True)
    df.drop(columns=['Embarked'])

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