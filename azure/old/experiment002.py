"""
# MS Azurez-ML Study

## Description

Learning how to use Azure-ML SDK by creating classification models for the Poker Hand Dataset

https://archive.ics.uci.edu/ml/datasets/Poker+Hand

## Author

Erick Medeiros Anastácio
2020-05-30

## Requirements

Python version
3.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from azureml.core import Run


def split_data(df):
    """
    Split the data into train and testing sets.

    Input

    - df: pandas dataframe

    Output

    A tuple containing:

    - train set
    - test set
    - train set classisications
    - test set classisications
    """
    # train test split
    X = df.copy()
    X.drop('class', axis=1, inplace=True)
    y = df['class'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=95276
        )

    return (X_train, X_test, y_train, y_test)


def trainer(data, model, model_name, grid_params):
    """
    Boiler plate for sklearn process.

    Input

    - data: tuple containing [X_train, X_test, y_train, y_test]
    - sklearn classification model object
    - model name
    - gridSearch parameters for the model

    Output

    - dictionary containing
        - model name
        - train and test sets scores (accuracy)
        - parameters

    """
    # data unpacking
    X_train, X_test, y_train, y_test = data

    # the grid
    grid = GridSearchCV(
        model, grid_params,
        scoring='accuracy',
        n_jobs=-1, cv=5
        )
    grid.fit(X_train, y_train)

    # the model
    model = grid.best_estimator_
    test_score = model.score(X_test, y_test)
    stats = {
        'model name': model_name,
        'train_score': grid.best_score_,
        'test_score': test_score,
        'parameters': model.get_params()
       }
    return stats

# Azure script
# azure experiment start
run = Run.get_context()

# the data from azure datasets/datastorage
df = run.input_datasets['poker'].to_pandas_dataframe()

# get a tuple containing train and testing sets
data = split_data(df)


# Random Forest
model = RandomForestClassifier(random_state=95276)
model_name = 'Random Forest'
grid_params = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'max_features': [3, 5, 10, 15, None],
    'min_samples_leaf': [2, 5],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None],
    'n_estimators': [500]
    }

results = trainer(data, model, model_name, grid_params)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=95276)
model_name = 'Gradient Boosting'
grid_params = {
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5],
    'n_estimators': [500]
    }

results = trainer(data, model, model_name, grid_params)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# azure finish
run.complete()
