"""
Project DP-100 Certification.

Description:
Learning how to use Azure-ML SDK

Creating classification models for the
Poker Hand Dataset
https://archive.ics.uci.edu/ml/datasets/Poker+Hand

Author:
Erick Medeiros Anastácio
2020-05-25

Python version:
3.7
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression

from azure.core import Run


# azure experiment start
run = Run.get_context()
# the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data'
df = pd.read_csv(url)

# set columns names, according to
# https://archive.ics.uci.edu/ml/datasets/Poker+Hand
col_names = []
for number in range(1,6):
    for name in ['Suit_', 'Rank_']:
        col = f'{name}{number}'
        col_names.append(col)

col_names.append('class')
df.columns = col_names

# train test split
X = df.copy()
X.drop('class', axis=1, inplace=True)
y = df['class'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42
    )


def trainer(model, model_name):
    """
    Boiler plate for sklearn.

    Returns model name, scores and parameters, as dict.
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


# Logistic Regression model
model = LogisticRegression()
model_name = 'logistic regression'
results = trainer(model, model_name)
run.log(model_name, results)

# Decision tree model
model = DecisionTreeClassifier()
model_name = 'decision tree'
results = trainer(model, model_name)
run.log(model_name, results)

# azure finish
run.complete()
