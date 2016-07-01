## V3 Classification

The classification architecture is an implementation of http://arxiv.org/abs/1512.00567 

### Data Format
The data needs to be in a tfrecords file(s). See the inputs [README](../../inputs/README.md) for more information.


### Directory Structure
Setup a directory with the following contents

- checkpoints/ : this is a directory for storing the model snapshots.
- summary/ : this is a directory for storing event logs that will be used by Tensorboard
- train.tfrecords : this is a tfrecords file that contains your training data.
- test.tfrecords : this is a tfrecords file that contains your testing data.
- config.yaml : this is a configuration file.

### Configuration
The configuration file is a yaml formatted file that specifies the hyperparameters of the model. See the example [configuration file](config.yaml.example). 

### Training
You can start the training process through the command line:
```shell
python v3/classification/start_test.py \
--tfrecords <path_to_directory>/train.tfrecords \
--log_dir <path_to_directory>/ \
--config <path_to_directory>/config.yaml \
--first_iteration
```

### Testing
You can start the testing process through the command line:
```shell
python v3/classification/start_test.py \
--tfrecords <path_to_directory>/test.tfrecords \
--checkpoint_dir <path_to_directory>/checkpoints \
--config <path_to_directory>/config.yaml
```
