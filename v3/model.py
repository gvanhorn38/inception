import tensorflow as tf

from network_utils import add_avg_pool, add_max_pool, add_conv

# Default Values
# WD : weight decay (regularization term)


# Figure 5 configuration
# This isn't exactly figure 5. There is a 5x5 filter bank here.
# So its like a combo of Figure 4 and Figure 5
def add_figure5(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_conv_1_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_conv_2_shape,
  tower_2_conv_shape,
  cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape = conv_shape, # [1, 1, 192, 64],    # [1, 1, 256, 64],   # [1, 1, 288, 64]
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):

    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape = tower_conv_shape, #[1, 1, 192, 48],   # [1, 1, 256, 48],    # [1, 1, 288, 48]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape = tower_conv_1_shape, #[5, 5, 48, 64],  # [5, 5, 48, 64],    # [5, 5, 48, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):

    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape = tower_1_conv_shape, #[1, 1, 192, 64],   # [1, 1, 256, 64],    # [1, 1, 288, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape = tower_1_conv_1_shape, #[3, 3, 64, 96],  # [3, 3, 64, 96] ,   # [3, 3, 64, 96]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape = tower_1_conv_2_shape, #[3, 3, 96, 96],  # [3, 3, 96, 96] ,   # [3, 3, 96, 96]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_2
  with graph.name_scope("tower"):

    tower_2_pool = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 3, 3, 1],  # [1, 3, 3, 1]   # [1, 3, 3, 1]
      strides=[1, 1, 1, 1],
      padding = "SAME",
      name="pool"
    )

    tower_2_conv = add_conv(
      graph = graph,
      input = tower_2_pool,
      shape = tower_2_conv_shape, #[1, 1, 192, 32],   # [1, 1, 256, 64],  # [1, 1, 288, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_1, tower_1_conv_2, tower_2_conv],
    name="join"
  )


# First grid size reduction
# mixed_3
def add_figure10_1(graph, input, cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape =  [3, 3, 288, 384],
    strides = [1, 2, 2, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 288, 64],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  [3, 3, 64, 96],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_2 = add_conv(
      graph = graph,
      input = tower_conv_1,
      shape =  [3, 3, 96, 96],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  pool = add_max_pool(
    graph=graph,
    input=input,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_2, pool],
    name="join"
  )


def add_figure6(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_conv_1_shape,
  tower_conv_2_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_conv_2_shape,
  tower_1_conv_3_shape,
  tower_1_conv_4_shape,
  tower_2_conv_shape,
  cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape =  conv_shape,
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  tower_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_2 = add_conv(
      graph = graph,
      input = tower_conv_1,
      shape =  tower_conv_2_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_1_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  tower_1_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape =  tower_1_conv_2_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_3 = add_conv(
      graph = graph,
      input = tower_1_conv_2,
      shape =  tower_1_conv_3_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_4 = add_conv(
      graph = graph,
      input = tower_1_conv_3,
      shape =  tower_1_conv_4_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_2
  with graph.name_scope("tower"):

    tower_2_pool = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 3, 3, 1],
      strides=[1, 1, 1, 1],
      padding = "SAME",
      name="pool"
    )

    tower_2_conv= add_conv(
      graph = graph,
      input = tower_2_pool,
      shape =  tower_2_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_2, tower_1_conv_4, tower_2_conv],
    name="join"
  )

# Second grid size reduction
# mixed_8
def add_figure10_2(graph, input, cfg):

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 768, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  [3, 3, 192, 320],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 768, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  [1, 7, 192, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape =  [7, 1, 192, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_3 = add_conv(
      graph = graph,
      input = tower_1_conv_2,
      shape =  [3, 3, 192, 192],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # pool
  pool = add_max_pool(
    graph=graph,
    input=input,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  return tf.concat(
    concat_dim=3,
    values = [tower_conv_1, tower_1_conv_3, pool],
    name="join"
  )

def add_figure7(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_mixed_conv_shape,
  tower_mixed_conv_1_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_mixed_conv_shape,
  tower_1_mixed_conv_1_shape,
  tower_2_conv_shape,
  use_avg_pool,
  cfg):

  # conv
  conv = add_conv(
    graph = graph,
    input = input,
    shape =  conv_shape,
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    with graph.name_scope("mixed"):
      tower_mixed_conv = add_conv(
        graph = graph,
        input = tower_conv,
        shape =  tower_mixed_conv_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

      tower_mixed_conv_1 = add_conv(
        graph = graph,
        input = tower_conv,
        shape =  tower_mixed_conv_1_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_1_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  tower_1_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    with graph.name_scope("mixed"):
      tower_1_mixed_conv = add_conv(
        graph = graph,
        input = tower_1_conv_1,
        shape =  tower_1_mixed_conv_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

      tower_1_mixed_conv_1 = add_conv(
        graph = graph,
        input = tower_1_conv_1,
        shape =  tower_1_mixed_conv_1_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

  # tower_2
  with graph.name_scope("tower"):
    if use_avg_pool:
      tower_2_pool = add_avg_pool(
          graph=graph,
          input=input,
          ksize=[1, 3, 3, 1],
          strides=[1, 1, 1, 1],
          padding = "SAME",
          name="pool"
        )
    else:
      tower_2_pool = add_max_pool(
        graph=graph,
        input=input,
        ksize=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        padding = "SAME",
        name="pool"
      )

    tower_2_conv = add_conv(
      graph = graph,
      input = tower_2_pool,
      shape =  tower_2_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_mixed_conv, tower_mixed_conv_1, tower_1_mixed_conv, tower_1_mixed_conv_1, tower_2_conv],
    name="join"
  )

# Rather than passing the graph around, we could do `with graph.as_default():`
def build(graph, input, cfg):

  # conv
  conv = add_conv(
    graph = graph,
    input = input,
    shape = [3, 3, 3, 32],
    strides = [1, 2, 2, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_1
  conv_1 = add_conv(
    graph = graph,
    input = conv,
    shape = [3, 3, 32, 32],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_2
  conv_2 = add_conv(
    graph = graph,
    input = conv_1,
    shape = [3, 3, 32, 64],
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # pool
  pool = add_max_pool(
    graph=graph,
    input=conv_2,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  # conv_3
  conv_3 = add_conv(
    graph = graph,
    input = pool,
    shape = [1, 1, 64, 80],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_4
  conv_4 = add_conv(
    graph = graph,
    input = conv_3,
    shape = [3, 3, 80, 192],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # pool_1
  pool_1 = add_max_pool(
    graph=graph,
    input=conv_4,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  #################
  # First block of inception modules
  # 3 modules as specified in Figure 5

  # mixed
  with graph.name_scope("mixed"):
    mixed = add_figure5(graph, pool_1,
      conv_shape = [1, 1, 192, 64],
      tower_conv_shape = [1, 1, 192, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 192, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 192, 32],
      cfg=cfg
    )

  # mixed_1
  with graph.name_scope("mixed"):
    mixed_1 = add_figure5(graph, mixed,
      conv_shape = [1, 1, 256, 64],
      tower_conv_shape = [1, 1, 256, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 256, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 256, 64],
      cfg=cfg
    )


  # mixed_2
  with graph.name_scope("mixed"):
    mixed_2 = add_figure5(graph, mixed_1,
      conv_shape = [1, 1, 288, 64],
      tower_conv_shape = [1, 1, 288, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 288, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 288, 64],
      cfg=cfg
    )

  # End first block of inception modules
  #################

  # First Inception module for grid size reduction
  with graph.name_scope("mixed"):
    mixed_3 = add_figure10_1(graph, mixed_2, cfg=cfg)

  # Second block of inception modules
  # 4 modules as specified in Figure 6
  # NOTE: rather than 5 modules as specified in the paper, there are only 4 modules in
  # the graph def
  # mixed_4
  with graph.name_scope("mixed"):
    mixed_4 = add_figure6(graph, mixed_3,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 128],
      tower_conv_1_shape =    [1, 7, 128, 128],
      tower_conv_2_shape =    [7, 1, 128, 192],
      tower_1_conv_shape =    [1, 1, 768, 128],
      tower_1_conv_1_shape =  [7, 1, 128, 128],
      tower_1_conv_2_shape =  [1, 7, 128, 128],
      tower_1_conv_3_shape =  [7, 1, 128, 128],
      tower_1_conv_4_shape =  [1, 7, 128, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_5
  with graph.name_scope("mixed"):
    mixed_5 = add_figure6(graph, mixed_4,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 160],
      tower_conv_1_shape =    [1, 7, 160, 160],
      tower_conv_2_shape =    [7, 1, 160, 192],
      tower_1_conv_shape =    [1, 1, 768, 160],
      tower_1_conv_1_shape =  [7, 1, 160, 160],
      tower_1_conv_2_shape =  [1, 7, 160, 160],
      tower_1_conv_3_shape =  [7, 1, 160, 160],
      tower_1_conv_4_shape =  [1, 7, 160, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_6
  with graph.name_scope("mixed"):
    mixed_6 = add_figure6(graph, mixed_5,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 160],
      tower_conv_1_shape =    [1, 7, 160, 160],
      tower_conv_2_shape =    [7, 1, 160, 192],
      tower_1_conv_shape =    [1, 1, 768, 160],
      tower_1_conv_1_shape =  [7, 1, 160, 160],
      tower_1_conv_2_shape =  [1, 7, 160, 160],
      tower_1_conv_3_shape =  [7, 1, 160, 160],
      tower_1_conv_4_shape =  [1, 7, 160, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_7
  with graph.name_scope("mixed"):
    mixed_7 = add_figure6(graph, mixed_6,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 192],
      tower_conv_1_shape =    [1, 7, 192, 192],
      tower_conv_2_shape =    [7, 1, 192, 192],
      tower_1_conv_shape =    [1, 1, 768, 192],
      tower_1_conv_1_shape =  [7, 1, 192, 192],
      tower_1_conv_2_shape =  [1, 7, 192, 192],
      tower_1_conv_3_shape =  [7, 1, 192, 192],
      tower_1_conv_4_shape =  [1, 7, 192, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # End second block of inception modules
  #################

  # Second Inception module for grid size reduction
  with graph.name_scope("mixed"):
    mixed_8 = add_figure10_2(graph, mixed_7, cfg=cfg)


  #################
  # Third block of inception modules
  # 2 modules as specified in Figure 7
  # mixed_9
  with graph.name_scope("mixed"):
    mixed_9 = add_figure7(graph, mixed_8,
      conv_shape                  = [1, 1, 1280, 320],
      tower_conv_shape            = [1, 1, 1280, 384],
      tower_mixed_conv_shape      = [1, 3, 384, 384],
      tower_mixed_conv_1_shape    = [3, 1, 384, 384],
      tower_1_conv_shape          = [1, 1, 1280, 448],
      tower_1_conv_1_shape        = [3, 3, 448, 384],
      tower_1_mixed_conv_shape    = [1, 3, 384, 384],
      tower_1_mixed_conv_1_shape  = [3, 1, 384, 384],
      tower_2_conv_shape          = [1, 1, 1280, 192],
      use_avg_pool = True,
      cfg=cfg
    )

  with graph.name_scope("mixed"):
    mixed_10 = add_figure7(graph, mixed_9,
      conv_shape                  = [1, 1, 2048, 320],
      tower_conv_shape            = [1, 1, 2048, 384],
      tower_mixed_conv_shape      = [1, 3, 384, 384],
      tower_mixed_conv_1_shape    = [3, 1, 384, 384],
      tower_1_conv_shape          = [1, 1, 2048, 448],
      tower_1_conv_1_shape        = [3, 3, 448, 384],
      tower_1_mixed_conv_shape    = [1, 3, 384, 384],
      tower_1_mixed_conv_1_shape  = [3, 1, 384, 384],
      tower_2_conv_shape          = [1, 1, 2048, 192],
      use_avg_pool = False, # use a max pool
      cfg=cfg
    )

  return mixed_10


  