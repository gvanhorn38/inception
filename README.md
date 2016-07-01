# Inception
Inception network archtectures for classification and detection. 
 
## V3 Classification

The classification architecture is an implementation of http://arxiv.org/abs/1512.00567 

### Data Format
The data needs to be in a tfrecords file with the following fields:
| Key | Value|
|-----|------|
|path | a full file path for the image|
|label| an integer in in the range [0, the total number of classes)|

You'll need at least 2 of these tfrecord files. One of them should contain your training data (train.tfrecords) and the other should contain testing data (test.tfrecords) 

### Directory Structure
Setup a directory with the following contents

- checkpoints/ : this is a directory for storing the model snapshots.
- summary/ : this is a directory for storing event logs that will be used by Tensorboard
- train.tfrecords : this is a tfrecords file that contains your training data.
- test.tfrecords : this is a tfrecords file that contains your testing data.
- config.yaml : this is a configuration file.

### Configuration
The configuration file is a yaml formatted file that specifies the hyperparameters of the model. See the example [configuration file](v3/classification/config.yaml.example). 

### Training
You can start the training process through the command line:
```sh
python v3/classification/start_test.py \
--tfrecords <path_to_directory>/train.tfrecords \
--log_dir <path_to_directory>/ \
--config <path_to_directory>/config.yaml \
--first_iteration
```

### Testing
You can start the testing process through the command line:
```sh
python v3/classification/start_test.py \
--tfrecords <path_to_directory>/test.tfrecords \
--checkpoint_dir <path_to_directory>/checkpoints \
--config <path_to_directory>/config.yaml
```

## V3 Detection