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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression

from azure.core import Run


def dataset_etl(url):
    """
    Run a quick ETL process.

    Input

    - url: data location

    Output

    A tuple containing:

    - train set
    - test set
    - train set classisications
    - test set classisications
    """
    df = pd.read_csv(url)

    # data check
    print('Shape: ', df.shape)
    print('Missing data:\n', df.isna().sum())

    # set columns names, according to
    # https://archive.ics.uci.edu/ml/datasets/Poker+Hand
    col_names = []
    for number in range(1,6):
        for name in ['Suit_', 'Rank_']:
            col = f'{name}{number}'
            col_names.append(col)

    col_names.append('class')
    df.columns = col_names

    # transform suits into categories
    suit_cols = [col for col in df.columns if col.startswith('Suit_')]
    df_trans = pd.get_dummies(df, columns=suit_cols, drop_first=True)

    # train test split
    X = df_trans.copy()
    X.drop('class', axis=1, inplace=True)
    y = df['class'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=95276
        )

    return X_train, X_test, y_train, y_test


def trainer(model, model_name):
    """
    Boiler plate for sklearn.

    Input

    - sklearn classification model object

    Output

    - dictionary containing
        - model name
        - train and test sets scores (accuracy)
        - parameters

    """
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    stats = {
        'model name': model_name,
        'train_score': train_score,
        'test_score': test_score,
        'parameters': model.get_params()
       }
    return stats


# Azure script
# azure experiment start
run = Run.get_context()

# the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data'

X_train, X_test, y_train, y_test = dataset_etl(url)

# Logistic Regression model
model = LogisticRegression(random_state=95276)
model_name = 'logistic regression'
results = trainer(model, model_name)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# Decision tree model
model = DecisionTreeClassifier(random_state=95276)
model_name = 'decision tree'
results = trainer(model, model_name)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)

# azure finish
run.complete()
