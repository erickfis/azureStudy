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
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from azureml.core import Run

import exp_resources as res


# Azure script
# azure experiment start
run = Run.get_context()

# the data from azure datasets/datastorage
df = run.input_datasets['poker'].to_pandas_dataframe()

# get a tuple containing train and testing sets
data = res.split_data(df)

# get parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--chosen_model',
    type=str,
    dest='chosen_model',
    default='logistic'
    )
args = parser.parse_args()
chosen_model = args.chosen_model

# run models
if chosen_model == 'tree':
    model = DecisionTreeClassifier(random_state=95276)
    model_name = 'decision tree'
    grid_params = {
        'max_depth': [3, 5, 10, 15, 20, None],
        'max_features': [3, 5, 10, 15, None],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None],
        }
elif chosen_model == 'rf':
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
elif chosen_model == 'gb':
    model = GradientBoostingClassifier(random_state=95276)
    model_name = 'Gradient Boosting'
    grid_params = {
        'min_samples_leaf': [1, 2, 5],
        'min_samples_split': [2, 5],
        'n_estimators': [500]
        }
else:
    model = LogisticRegression(random_state=95276)
    model_name = 'logistic regression'
    grid_params = {
        'C': [.001, .01, 1, 10],
        'class_weight': ['balanced', None],
        'max_iter': [1000]
        }

# anyway
print(f'Running {model_name}')
results = res.trainer(data, model, model_name, grid_params)
print('model name:', results['model name'])
print(f'test_score (accuracy): {results["test_score"]:.4f}')
run.log(model_name, results)
run.log('accuracy', results['test_score'])

# azure finish
run.complete()
