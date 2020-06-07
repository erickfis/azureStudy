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

import numpy as np
import pandas as pd
from azureml.core.model import Model
from azureml.core import Run
# import azureml.train.automl

import exp_resources as res
import joblib

# Azure script
# azure experiment start
run = Run.get_context()

# the data from azure datasets/datastorage
df = run.input_datasets['poker'].to_pandas_dataframe()

# get a tuple containing train and testing sets
X_train, X_test, y_train, y_test = res.split_data(df)

# Get the path to the deployed model file and load it
model_path = Model.get_model_path('model_automl')
model = joblib.load(model_path)

predictions = model.predict(X_test)
acc = (predictions == y_test).mean()
print(acc)

run.log('accuracy', acc)

# azure finish
run.complete()
