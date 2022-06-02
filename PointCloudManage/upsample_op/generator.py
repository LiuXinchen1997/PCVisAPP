import tensorflow as tf

from PointCloudManage.upsample_op.ops import recurrent_context_encoding, hierarchical_feature_expansion, conv1d

class Generator(object):
    def __init__(self, cfg, is_training, name="Displacement"):
        self.cfg = cfg
        self.is_training = is_training
        self.name = name
        self.reuse = tf.AUTO_REUSE
        self.num_point = self.cfg.patch_num_points
        self.up_ratio = self.cfg.up_ratio
        self.up_ratio_real = self.up_ratio + self.cfg.more_up
        self.out_num_point = int(self.num_point*self.up_ratio)

    def __call__(self, inputs):
        use_bn = False
        use_ibn = False

        with tf.variable_scope(self.name, reuse=self.reuse):
            # inputs = gather_point(inputs, farthest_point_sample(self.num_point, inputs))
            B = inputs.get_shape()[0].value
            N = inputs.get_shape()[1].value
            C = inputs.get_shape()[2].value

            point_features = recurrent_context_encoding(inputs, scope="recurrent_context_encoding",
                                                                is_training=self.is_training, bn_decay=None)

            coarse, coarse_feat = hierarchical_feature_expansion(point_features, scope="hierarchical_feature_expansion",
                                                    is_training=self.is_training, bn_decay=None)

        with tf.variable_scope(self.name + "/refine", reuse=self.reuse):
            fine_feat = recurrent_context_encoding(coarse, scope="recurrent_context_encoding",
                                                           is_training=self.is_training, bn_decay=None, knn_list=[4, 8])
            fine_feat = tf.concat([fine_feat, coarse_feat], axis=-1)

            fine_feat = conv1d(fine_feat, 128, 1,
                                   padding='VALID', scope='layer3_prep', is_training=self.is_training, bn=use_bn,
                                   ibn=use_ibn,
                                   bn_decay=None)

            fine_feat = conv1d(fine_feat, 64, 1,
                                   padding='VALID', scope='layer4_prep', is_training=self.is_training, bn=use_bn,
                                   ibn=use_ibn,
                                   bn_decay=None)

            fine = conv1d(fine_feat, 3, 1,
                              padding='VALID', scope='layer5_prep', is_training=self.is_training, bn=use_bn,
                              ibn=use_ibn,
                              bn_decay=None, activation_fn=None)

            fine = coarse + fine

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return coarse, fine
