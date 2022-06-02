import tensorflow as tf
import numpy as np
import os
from time import time

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from PointCloudManage.upsample_op.generator import Generator
from PointCloudManage.upsample_op.tf_ops.sampling.tf_sampling import farthest_point_sample


class Model(object):
    def __init__(self, cfg, sess):
        self.cfg = cfg
        self.sess = sess
    
    @staticmethod
    def _pre_load_checkpoint(checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
            epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            return epoch_step, ckpt.model_checkpoint_path
        else:
            return 0, None
    
    @staticmethod
    def _extract_knn_patch(queries, pc, k):
        """
        queries [M, C]
        pc [P, C]
        """
        knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn_search.fit(pc)
        knn_idx = knn_search.kneighbors(queries, return_distance=False)
        k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
        return k_patches
    
    @staticmethod
    def _normalize_points(points):
        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid
        furthest_distance = np.amax(
            np.sqrt(np.sum(points ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)
        points = points / furthest_distance
        return points, centroid, furthest_distance

    def patch_prediction(self, patch_point):
        # normalize the point clouds
        patch_point, centroid, furthest_distance = Model._normalize_points(patch_point)
        patch_point = np.expand_dims(patch_point, axis=0)
        pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point})
        pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
        return pred

    def pc_prediction(self, pc):
        ## get patch seed from farthestsampling
        points = tf.convert_to_tensor(np.expand_dims(pc, axis=0), dtype=tf.float32)
        start = time()
        print('------------------patch_num_point:', self.cfg.patch_num_points)
        seed1_num = int(pc.shape[0] / self.cfg.patch_num_points * self.cfg.patch_num_ratio)

        ## FPS sampling
        seed = farthest_point_sample(seed1_num, points).eval()[0]
        seed_list = seed[:seed1_num]
        print("farthest distance sampling cost", time() - start)
        print("number of patches: %d" % len(seed_list))
        input_list = []
        up_point_list = []

        patches = Model._extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.cfg.patch_num_points)

        for i, point in enumerate(patches):
            if self.cfg.progress_record:
                prog_val = str(int((i + 1) / len(patches) * 100))
                with open(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'mypcbase', 'temp', 'prog-upsample.txt'), 'w') as f:
                    f.write(prog_val)

            up_point = self.patch_prediction(point)
            up_point = np.squeeze(up_point, axis=0)
            input_list.append(point)
            up_point_list.append(up_point)

        return input_list, up_point_list

    def test(self, points):
        self.inputs = tf.placeholder(tf.float32, shape=[1, self.cfg.patch_num_points, 3])
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        Gen = Generator(self.cfg, is_training, name='generator')
        _, self.pred_pc = Gen(self.inputs)

        saver = tf.train.Saver()
        restore_epoch, checkpoint_path = Model._pre_load_checkpoint(self.cfg.model_path)
        saver.restore(self.sess, checkpoint_path)
        
        start = time()
        num_points = points.shape[0]
        points, centroid, furthest_distance = Model._normalize_points(points)
        input_list, pred_list = self.pc_prediction(points)
        end = time()
        print("total time: ", end - start)
        pred_pc = np.concatenate(pred_list, axis=0)
        pred_pc = (pred_pc * furthest_distance) + centroid

        pred_pc = np.reshape(pred_pc, [-1, 3])
        idx = farthest_point_sample(num_points * self.cfg.up_ratio, pred_pc[np.newaxis, ...]).eval()[0]
        pred_pc = pred_pc[idx, 0:3]

        return pred_pc