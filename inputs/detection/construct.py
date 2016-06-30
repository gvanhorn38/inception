
import inputs

def construct_network_input_nodes(
    tfrecords,  
    **kwargs
  ):

  return inputs.input_nodes(tfrecords=tfrecords, **kwargs)