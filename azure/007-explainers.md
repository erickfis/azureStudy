# Explainers

- Global feature importance: The measure of influence of each feature for predictions on the test set.

- Local feature importance: The measure of influence of each feature for a particular prediction, for each class.

## Types of Explainers

- MimicExplainer: explainer targetted for a specific model
- TabularExplainer: explainer that automatically chooses (SHAP) the best algorithm to explain your model
- PFIExplainer: permutation - shuffles feature values and meausures its impact on the prediction.

## How to use

```python
# in the experiment script

from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient

explainer = TabularExplainer(
    model,
    features=feature_columns_list,
    classes=labels_column_list
    )
explanation = explainer.explain_global(X_test)
```
