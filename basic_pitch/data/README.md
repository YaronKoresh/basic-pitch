# Data / Training
The code and scripts in this section deal with training basic pitch on your own. Scripts in the `datasets` folder allow one to download and process a selection of the datasets used to train the original model. Each of these download scripts has the following keyword arguments:
* **--source**: Source directory to download raw data to. It defaults to `$HOME/mir_datasets/{dataset_name}`.
* **--destination**: Directory to write processed data to. It defaults to `$HOME/data/basic_pitch/{dataset_name}`.
* **--runner**: The method used to run the Beam Pipeline for processing the dataset. Options include `DirectRunner`, running directly in the code process running the pipeline, `PortableRunner`, which can be used to run the pipeline in a docker container locally, and `DataflowRunner`, which can be used to run the pipeline in a docker container on Dataflow. 
* **--timestamped**: If passed, the dataset will be put into a timestamp directory instead of 'splits'.
* **--batch-size**: Number of examples per tfrecord when partitioning the dataset.
* **--sdk_container_image**: The Docker container image used to process the data if using `PortableRunner`.
* **--job_endpoint**: the endpoint where the job is running. It defaults to `embed` which works for `PortableRunner`. 

The scripts of the datasets that split their tracks themselves (`guitarset`, `ikala` and `medleydb_pitch`) also have these arguments:
* **--train-percent**: Fraction of the tracks to mark as train. It defaults to `0.8`.
* **--validation-percent**: Fraction of the tracks to mark as validation. It defaults to `0.1`, and only `guitarset` uses it, where the remaining tracks are the test split. The train and validation fractions must add up to less than 1. `ikala` and `medleydb_pitch` mark all tracks that are not train as validation.
* **--split-seed**: Seed of the random number generator used for the split, to get the same split again. It defaults to a random seed.

Additional arguments that work with Beam in general can be used as well, and will be passed along and used by the pipeline. If using `DataflowRunner`, you will be required to pass `--temp_location={Path to GCS Bucket}`, `--staging_location={Path to GCS Bucket}`, `--project={Name of GCS Project}` and `--region={GCS region}`. 

## Training
Once the datasets are processed, `bp-train` trains the model on them. It needs the `tf` extra (`pip install 'basic-pitch[tf]'`). `--source` is the directory that holds the processed datasets, which is the parent of the `{dataset_name}/splits` directories, and `--output` is the directory the trained model, its checkpoints and the tensorboard logs are saved in. Pick the datasets to train on with `--maestro`, `--guitarset`, `--medleydb-pitch`, `--ikala` and `--slakh`; at least one is required:
```bash
bp-train --source $HOME/data/basic_pitch --output $HOME/models/basic_pitch --guitarset
```
Run `bp-train --help` for the other options, such as `--epochs`, `--batch-size`, `--learning-rate` and `--no-contours`.
