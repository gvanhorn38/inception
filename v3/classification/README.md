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

| Key | Value |
|-----|-------|
| RANDOM_SEED | Specify the value for the random seed. Use this to help reproduce experiments. |
| SESSION_CONFIG | This dictionary will be used to configure the TensorFlow session object.|
| SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION |  The amount of gpu memory to reserve.  | 
| FIXED_BASE | If true, then the base network parameters will be held fixed. Only the "new" layers (softmax, etc.) will be trainable. If you have a small dataset, then this should probably be set to true. |
| NUM_CLASSES | The number of classes in this task. |
| INPUT_SIZE | The input size. The input will be square INPUT_SIZE x INPUT_SIZE. |
| IMAGE_MEAN | The mean value of an image (across all channels). |
| IMAGE_STD | The standard deviation of an image (across all channels). |
| INPUT_TYPE | The format of the input to the network. To send the whole image into the network, use `whole_image_input`. If bounding boxes are present in the input data, then you can use `bbox_input` to extract bounding boxes.  |
| MAINTAIN_ASPECT_RATIO | Inputs to the network must be square, and will have dimensions INPUT_SIZE x INPUT_SIZE. When resizing the input, we can either maintain the aspect ratio and pad with 0s, or we can ignore the aspect ratio and squish the images into the proper shape. |
| AUGMENT_IMAGE | Augment the images prior to sending them through the network? This is an easy way of synthetically boosting the amount of training data. |
| IMAGE_AUGMENTATIONS | A dictionary of augmentation parameters. |
| IMAGE_AUGMENTATIONS.CROP | Take a random crop from the image.  |
| IMAGE_AUGMENTATIONS.CROP_UPSCALE_FACTOR | When cropping the image, first scale it up by this amount, then crop. |
| IMAGE_AUGMENTATIONS.FLIP_LEFT_RIGHT | Randomly flip the image across the middle.  |
| IMAGE_AUGMENTATIONS.BRIGHTNESS | Randomly adjust the brightness of an image. |
| IMAGE_AUGMENTATIONS.CONTRAST | Randomly adjust the contrast of an image. |
| NUM_INPUT_THREADS | The number of threads that will be used to fill the input queues. |
| BATCH_SIZE | The number of images to batch together for a single pass. Typically you want this to be as large as possible, before running out of gpu memory. |
| NUM_TRAIN_ITERATIONS | The maximum number of training iterations to run. |
| LEARNING_RATE | The initial learning rate. |
| LEARNING_RATE_DECAY_STEPS | The number of steps between decaying the learning rate. |
| LEARNING_RATE_DECAY |  The learning rate decay factor.  |
| LEARNING_RATE_STAIRCASE | Force the learning rate to decay in a staircase fashion (integer division vs float).  |
| RMSPROP_DECAY | Discounting factor for the history/coming gradient. |
| RMSPROP_EPSILON |  Small value to avoid zero denominator. |
| MOVING_AVERAGE_DECAY | Decay factor for maintaining the moving averages. Should be close to 1. |
| SAVE_EVERY_N_ITERATIONS | The number of iterations before snapshoting the model.  |
| USE_EXTRA_CLASSIFICATION_HEAD | An extra classification head can be added to speed up training time.  |
| USE_THIRD_CLASSIFICATION_HEAD | A third classification head can also be added. The benefit of this is unkown. |


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
