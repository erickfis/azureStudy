## Experiments

A named process, like running a  script or pipeline that generate metrics and which can be tracked.

- explore data
- build & evaluate predictive models

### Logging

- log
- log_list
- log_row
- log_table
- log_image

    from azureml.core import Experiment

    # create an experiment variable
    experiment = Experiment(workspace = ws, name = "my-experiment")

    # start the experiment
    run = experiment.start_logging()

    # experiment code goes here
    var = 'my log'
    run.log('observations', var)

    # end the experiment
    run.complete()


### Retrieving log info

    from azureml.widgets import RunDetails
    RunDetails(run).show()

or

    import json
    metrics = run.get_metrics()
    print(json.dumps(metrics, indent=2))

### Output files

    import json
    files = run.get_file_names()
    print(json.dumps(files, indent=2))


### Running scripts as experiments

    from azure.core import Run
    run = Run.get_context()

    # code go here
    run.log('some info', info_var)

    # write to output
    data.to_csv('outputs/my_data.csv')
    run.complete()


### Defining a run configuration

To run scripts as experiments, we need a run config that defines the environment and a script run config that links both.


    from azure.core import Experiment, RunConfiguration, ScriptRunConfig

    # create run config obj
    run_config = RunConfiguration()

    # create a script config obj
    script_run_config = ScriptRunConfig(
        source_directory='scripts_folder',
        script='my_experiment.py',
        run_config=run_config
        )

    # submit
    exp = Experiment(workspace=ws, name='my exp')
    run = exp.submit(config=script_run_config)
    run.wait_for_completion(show_output=True)
