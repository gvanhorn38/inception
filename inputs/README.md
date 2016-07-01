# Inputs

The inputs to the classification and detection pipelines share a common format. The data needs to be stored in an [Example protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto). The protocol buffer will have the following fields:

| Key | Value |
|-----|-------|
| image/id | string containing an identifier for this image. |
| image/encoded | string containing JPEG encoded image in RGB colorspace|
| image/height | integer, image height in pixels |
| image/width | integer, image width in pixels |
| image/colorspace | string, specifying the colorspace, always 'RGB' |
| image/channels | integer, specifying the number of channels, always 3 |
| image/format | string, specifying the format, always'JPEG' |
| image/filename | string containing the basename of the image file |
| image/class/label | integer specifying the index in a classification layer. The label ranges from [0, num_labels), e.g 0-99 if there are 100 classes. |
|  image/class/text | string specifying the human-readable version of the label e.g. 'White-throated Sparrow' |
| image/object/bbox/xmin | a float array, the left edge of the bounding boxes; normalized coordinates. |
| image/object/bbox/xmax | a float array, the right edge of the bounding boxes; normalized coordinates. |
| image/object/bbox/ymin | a float array, the top left corner of the bounding boxes; normalized coordinates. |
| image/object/bbox/ymax | a float array, the top edge of the bounding boxes; normalized coordinates. |
| image/object/bbox/label | an integer array, specifying the index in a classification layer. The label ranges from [0, num_labels) |
| image/object/bbox/count | an integer, the number of bounding boxes | 

Note the bounding boxes are stored in normalized coordinates, meaning that the x values have been divided by the width of the image, and the y values have been divided by the height of the image. This ensures that the object location can be recovered on any (aspect-perserved) resized version of the original image.

The [inputs.create_tfrecords.py](create_tfrecords.py) file has a convience function for generating the tfrecord files. You will need to preprocess your dataset and get it into a python list of dicts. Each dict represents an image and should have the following format:

```python
{
  "filename" : "the full path to the image", 
  "id" : "an identifier for this image",
  "class" : {
    "label" : "integer in the range [0, num_classes)",
    "text" : "a human readable string for this class"
  },
  "object" : {
    "bbox" : {
      "xmin" : "an array of float values",
      "xmax" : "an array of float values",
      "ymin" : "an array of float values",
      "ymax" : "an array of float values",
      "label" : "an array of integer values, in the range [0, num_classes)"
    }
  }
}
```

Not all of the fields are required. For example, if you just want to train a classifier using the whole image as an input, then your dicts will need to have at least the following structure:
```python
{
  "filename" : "the full path to the image", 
  "id" : "an identifier for this image",
  "class" : {
    "label" : "integer in the range [0, num_classes)"
  }
}
```

Once you have your dataset preprocessed, you can use a method in [inputs.create_tfrecords.py](create_tfrecords.py) to create the tfrecords files. For example:

```python
train_dataset = None # this should be your array of dicts. Don't forget that 
                     #   you'll want to separate your training and testing data.

from inputs.create_tfrecords import create
create(
  dataset=train_dataset,
  dataset_name="train",
  output_directory="/home/gvanhorn/Desktop/train_dataset",
  num_shards=10,
  num_threads=5
)
```
