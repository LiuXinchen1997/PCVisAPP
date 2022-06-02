import tensorflow as tf
import numpy as np

from PointCloudManage.upsample_op.tf_ops.grouping.tf_grouping import knn_point_2


def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope, reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def cal_knn_idx(point_cloud, k=16):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    distance, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
    idx = idx[:, :, 1:, :]
    distance = distance[:, :, 1:]
    return idx, distance


def get_knn_feature(point_cloud, k=16, idx=None):
    cur_idx = idx[:, :, 0:k, :]
    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, cur_idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature


def dense_conv_idx(feature, n=3, growth_rate=64, k=16, idx=None, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = get_knn_feature(feature, k=k, idx=idx) # [B N K 2*C]

        y = conv2d(y, 256, [1, 1], padding='VALID', scope='l1', **kwargs)
        y = tf.reduce_max(y, axis=-2)
        return y, idx


def layer_fusion(layer1, layer2, feat_dim, scope='layer_fusion', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        aggregate = tf.concat([layer1, layer2], axis=-1)

        zt = conv1d(aggregate, 128, 1, padding='VALID', scope='layer0_0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)

        zt = tf.sigmoid(conv1d(zt, feat_dim, 1, padding='VALID', scope='layer0_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None))

        rt = conv1d(aggregate, 128, 1,
                    padding='VALID', scope='layer1_0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                    bn_decay=bn_decay)
        rt = tf.sigmoid(conv1d(rt, feat_dim, 1,
                               padding='VALID', scope='layer1_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                               bn_decay=bn_decay, activation_fn=None))

        ht_bar = conv1d(tf.concat([tf.multiply(rt, layer1), layer2], axis=-1), 128, 1,
                        padding='VALID', scope='layer3_0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                        bn_decay=bn_decay)

        ht_bar = tf.tanh(conv1d(ht_bar, feat_dim, 1,
                                padding='VALID', scope='layer3_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                bn_decay=bn_decay, activation_fn=None))

        ht = tf.multiply(1-zt, layer1) + tf.multiply(zt, ht_bar)

        return ht


def recurrent_context_encoding(inputs, scope='recurrent_context_encoding', is_training=True, bn_decay=None, knn_list=[4, 8, 16, 32]):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        base_dim = 24
        layer_dim = 128
        comp = 32
        dense_n = 3
        use_bn = False
        use_ibn = False
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, base_dim, [1, 1],
                             padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)
        layer_features = l0_features
        feature_list = []
        knn_idx, _ = cal_knn_idx(layer_features, k=32)

        for i in range(len(knn_list)):
            layer_features, _ = dense_conv_idx(layer_features, growth_rate=comp, n=dense_n, k=knn_list[i], idx = knn_idx,
                                             scope="layer1_%d" % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                             bn_decay=bn_decay)
            layer_features = tf.expand_dims(layer_features, axis=2)
            layer_features = conv2d(layer_features, layer_dim, [1, 1],
                                 padding='VALID', scope='layer2_%d' % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                 bn_decay=bn_decay, activation_fn=None)
            layer_features = tf.squeeze(layer_features, axis=2)

            if i == 0:
                feature_list.append(layer_features)
            else:
                feature_fusion = layer_fusion(feature_list[-1], layer_features, layer_dim,
                                            scope='layer_fusion_%d' % i,
                                            is_training=is_training, bn_decay=bn_decay)
                feature_list.append(feature_fusion)

            layer_features = feature_list[-1]

    return feature_list[-1]


def expansion_grid(direction=0):
    if direction == 0:
        grid = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    else:
        grid = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    grid_tensor = tf.convert_to_tensor(grid)
    return grid_tensor


def dense_knn_idx(feature, n=3, growth_rate=64, k=16, idx=None, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = get_knn_feature(feature, k=k, idx=idx) # [B N K 2*C]
        y = conv2d(y, 256, [1, 1], padding='VALID', scope='l1' , **kwargs)
        return y


def distribution_attention(inputs, scope='distribution', is_training=True, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size = inputs.get_shape()[0].value
        num_points = inputs.get_shape()[1].value
        n_knn = [4, 8, 16, 32]
        layer_dim = 128
        comp = 32
        dense_n = 2
        n_step = 3
        use_bn = False
        use_ibn = False
        inputs = tf.squeeze(inputs, axis=2)
        knn_idx, knn_distance = cal_knn_idx(inputs, k=16)
        feature_list = []
        current_feature = inputs
        for i in range(n_step):
            layer_features = dense_knn_idx(current_feature, growth_rate=comp, n=dense_n, k=n_knn[i], idx=knn_idx,
                                           scope="layer_%d" % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                           bn_decay=bn_decay)

            attention = tf.nn.softmax(-knn_distance[:, :, :n_knn[i]], dim=2)
            attention = tf.expand_dims(attention, axis=3)
            att_feature = tf.reduce_sum(tf.multiply(attention, layer_features), axis=2)

            att_feature = conv2d(tf.expand_dims(att_feature, axis=2), layer_dim, [1, 1],
                                 padding='VALID', scope='layer_att_%d' % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                 bn_decay=bn_decay, activation_fn=None)
            if i == 0:
                feature_list.append(tf.squeeze(att_feature, axis=2))
            else:
                feature_fusion = layer_fusion(feature_list[-1], tf.squeeze(att_feature, axis=2), layer_dim,
                                            scope='layer_fusion', is_training=is_training, bn_decay=bn_decay)
                feature_list.append(feature_fusion)
                # current_feature = feature_fusion  # !!!
            current_feature = feature_list[-1]  # !!!

        final_feature = tf.expand_dims(feature_list[-1], axis=2)

        return final_feature


def hierarchical_feature_expansion(inputs, scope='expansion', is_training=True, bn_decay=None):
    batch_size = inputs.get_shape()[0].value
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        layer_dim = 128
        use_bn = False
        use_ibn = False
        up_steps = 2
        step_up_ratio = 2
        current_feature = tf.expand_dims(inputs, axis=2)
        for i in range(up_steps):
            cur_num_point = current_feature.get_shape()[1].value
            l0_features = conv2d(current_feature, layer_dim, [1, 1],
                                 padding='VALID', scope='expand_0_%d' % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                 bn_decay=bn_decay, activation_fn=None)
            grids = expansion_grid(direction=i)
            grids = tf.reshape(grids, [1, 1, 2, 2])
            grids = tf.tile(grids, [batch_size, cur_num_point, 1, 1])
            # lgrids = tf.get_variable("lgrid%d" % i, shape=[batch_size, cur_num_point, 2, 2], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0), dtype=tf.float32)

            l0_features = tf.tile(l0_features, [1, 1, 2, 1])
            l0_features = tf.concat([l0_features, grids], axis=-1)
            # l0_features = tf.concat([l0_features, lgrids], axis=-1)
            current_feature = conv2d(l0_features, layer_dim, [1, 1],
                                 padding='VALID', scope='expand_1_%d' % i, is_training=is_training, bn=use_bn, ibn=use_ibn,
                                 bn_decay=bn_decay, activation_fn=None)
            current_feature = tf.reshape(current_feature, [batch_size, cur_num_point*step_up_ratio, 1, layer_dim])
            current_feature = distribution_attention(current_feature, scope='distribution_%d' % i, is_training=is_training, bn_decay=bn_decay)

        l0_features = tf.squeeze(current_feature, axis=2)
        l1_fc_layer = conv1d(l0_features, 128, 1,
                             padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)

        l2_fc_layer = conv1d(l1_fc_layer, 64, 1,
                             padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay)

        l3_fc_layer = conv1d(l2_fc_layer, 3, 1,
                             padding='VALID', scope='layer5_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                             bn_decay=bn_decay, activation_fn=None)

        return l3_fc_layer, l0_features
