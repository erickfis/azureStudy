# Azure Auto ML

Azure ML provides model training for classification, regression and time series forecasting.

The total number of models to be built can be adjusted, as well the type of algorithms to be used

    AutoMLConfig(
        iterations=1000,
        whitelist_models=['KNN'],
        blacklist_models=['tree'],

    )


## Preprocessing

Scaling and Normalization will be applied automatically.

Other Preprocessing tasks are optional


## Submitting an AutoML experiment

    from azureml.core.compute import ComputeTarget
    from azureml.widgets import RunDetails
    from azureml.train.estimator import Estimator
    from azureml.train.automl import AutoMLConfig

    # get compute target and start it
    cpu_cluster = ComputeTarget(workspace=ws, name='pc3')
    cpu_cluster.start()
    cpu_cluster.wait_for_completion(show_output=True)


    # set the options
    automl_config = AutoMLConfig(
        name='PokerHand_Classification_AutoML',
        task='classification',
        compute_target=cpu_cluster,
        label_column_name='class',
        iterations=500, # 500 models will be built
        primary_metric = 'accuracy',
        featurization='auto',
        n_cross_validations=5,
        enable_early_stopping=True,
        max_cores_per_iteration=-1,
        experiment_timeout_hours=4,
        training_data=train_dataset,
        )

    # create exp and submit it
    automl_experiment = Experiment(ws, 'exp_name')
    automl_run = automl_experiment.submit(automl_config)
    RunDetails(automl_run).show()
    automl_run.wait_for_completion(show_output=True)


## Get metrics & best model

    best_run, fitted_model = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()
    for metric_name in best_run_metrics:
        metric = best_run_metrics[metric_name]
        print(metric_name, metric)


## Check Preprocessing steps

    for step in fitted_model.named_steps:
        print(step)

## Register the model


    from azureml.core import Model
    best_run.register_model(
        model_path='outputs/model.pkl',
        model_name='model_automl',
        tags={'Training context':'Auto ML'},
        properties={
            'AUC': best_run_metrics['AUC_weighted'],
            'Accuracy': best_run_metrics['accuracy']
            }
        )
