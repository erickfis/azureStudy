# Pipelines

Azure ML gives the option to divide a machine Learning experiment into small tasks and then to orchestrate their execution through pipelines.

Pipeline executions can be scheduled or triggered by client applications through a REST endpoint.

In Azure ML, each task is a step. The steps can be arranged sequentially or in parallel. Each step can be run on a specific compute target.

The steps can be any of the following:

- script
- estimator
- datatransfer
- databricks
- adla (SQL)


## Creating a pipeline

First create the steps:

    from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

    step1 = PythonScriptStep(...)
    step2 = EstimatorStep(...)

Now create the pipeline and run it as an experiment:

    from azureml.pipeline.core import Pipeline
    from azureml.core import experiment

    train_pipe = Pipeline(workspace=ws, steps=[step1, step2])

    exp = Experiment(workspace=ws, name='my_pipe_exp')
    pipe_run = exp.submit(train_pipe)


## Passing data between steps

Passing data between steps can be done through PipelineData, a intermediary store for data:

    step1 > PipelineData > step2


The notebook:

    from azureml.pipeline.core import PipelineData
    from azureml.pipeline.step import PythonScriptStep, EstimatorStep

    # get data from dataset
    raw_ds = Dataset.get_by_name(ws, 'my_data')

    # create a PipelineData
    data_store = ws.get_default_datastore()
    proc_data = PipelineData('processed', datastore=data_store)

    # script step
    step1 = PythonScriptStep(
        inputs=[raw_ds.as_named_input('raw_data')],
        outputs=[proc_data],
        arguments=['--folder', proc_data],
        ....
        )

    # estimator step
    step2 = EstimatorStep(
        inputs=[proc_data],
        estimator_entry_script_arguments=['--folder', proc_data],
        ...
        )

The experiment code:

    run = Run.get_context()

    # input data
    raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

    # output folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder')
    args = parser.parse_args()
    output_folder = args.folder

    # proc data
    proc_data = raw_df.groupby(....)
    ....

    # export to PipelineData
    os.makedirs(output_folder, exists_ok=True)
    path = os.path.join(output_folder, 'proc_data.csv')
    proc_data.to_csv(path)
