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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
        n_jobs=4, cv=5
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

# Logistic Regression model
model = LogisticRegression(random_state=95276)
model_name = 'logistic regression'
grid_params = {
    'C': [.001, .01, 1, 10],
    'class_weight': ['balanced', None],
    'max_iter': [500]
}
results = trainer(data, model, model_name, grid_params)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# Decision tree model
model = DecisionTreeClassifier(random_state=95276)
model_name = 'decision tree'
grid_params = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'max_features': [3, 5, 10, 15, None],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None],
    }
results = trainer(data, model, model_name, grid_params)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# azure finish
run.complete()
