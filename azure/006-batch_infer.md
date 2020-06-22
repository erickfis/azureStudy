# Batch inferencing

After training and registering our models, we can use them for batch inferencing tasks.

In order to run a batch inferencing task, we need to:

- register our models
- create a scoring script, containing init() and run() functions
- create a pipeline
- run the pipeline

## Registering the model

```python
from azureml.core import Model

model = Model.register(
    workspace=myws,
    model_name='model_name',
    model_path='model.pkl',
    description='model description'
    )

# or
run.register_model(...)
```
## Scoring script


```python
import os
import joblib
import numpy as np

from azureml.core import Model

def init():
    """Needed by the pipeline step."""
    global model

    model_path = Model.get_model_path('model_name')
    model = joblib.load(model_path)


def run(batches):
    """Called for each batch."""

    resultlist = []
    for batch in bathes:
        data = np.genfromtxt(batch, delimiter=',')
        pred = model.predict(data.reshape(1,-1))
        resultlist.append(pred)

    return resultlist

```

## The pipeline

```python
from azureml.core import Experiment
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.pipeline.core import PipelineData, Pipeline

# data
batch_data = ws.datasets('my_data')

# output
ds = ws.get_default_datastore()
output_dir = PipelineData(
    name='inferences',
    datastore=ds,
    output_path_on_compute='results'
    )


# step config
parallel_run_config = ParallelRunConfig(...)

# The step
parallel_step = ParallelRunStep(...)

# the pipe
pile = Pipeline(workspace=ws, steps=[parallel_step])

# Run the pipeline
pipe_run = Experiment(ws, 'batch_pred_pipe')
pipe_run.submit(pipeline)

pipe_run.wait_for_completion(show_output=True)



```
