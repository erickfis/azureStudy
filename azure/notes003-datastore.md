## DataStores

There are different types of DataStores available:

- Azure Storage Blob
- Azure Storage file container
- Azure Data Lake Storage
- Azure SQL Database
- Azure Databricks file system

Every workspace has:

- Azure Storage Blob
- Azure Storage fie container

To create a reference to a default datastore:

    default_store = ws.get_default_datastore()


To upload data:

    blob_ds.upload(src_dir='/files',
                   target_path='/data/files',
                   overwrite=True, show_progress=True)


## Datasets

Datasets are versioned data packages that can be used in experiments. There are two types of datasets:
- tabular
- file

To create a tabular dataset:

    from azureml.core import Dataset
    csv_path = [(default_store, 'my_table.csv')]
    tab_ds = Dataset.Tabular.from_delimited_files(csv_path)
    tab_ds = tab_ds.register(workspace=ws, name='my_table', create_new_version=1)

To retrieve a dataset:

    ds1 = ws.datasets['my_table']
    # or
    ds1 = Dataset.get_by_name(workspace='ws', name='my_table', version=1)

## Passing a dataset to a experiment script

First set the estimator:

    estimator = SKLearn(
        source_directory='experiment_folder',
        entry_script='training_script.py',
        compute_target='local',
        inputs=[tab_ds.as_named_input('csv_data')],
        pip_packages=['azureml-dataprep[pandas]'
        )

Then use it in the script:

    data = run.input_datasets['csv_data'].to_pandas_dataframe()
