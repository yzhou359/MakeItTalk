import scipy.io
import scipy.misc
from glob import glob
import os
import numpy as np
from image_utils import *
from ops import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import contrib


class DeepHeatmapsModel(object):

    """facial landmark localization Network"""

    def __init__(self, mode='TRAIN', train_iter=500000, learning_rate=1e-8, image_size=256, c_dim=3, batch_size=10,
                 num_landmarks=68, augment=True, img_path='data', save_log_path='logs', save_sample_path='sample',
                 save_model_path='model',test_model_path='model/deep_heatmaps_primary-1000'):

        self.mode = mode
        self.train_iter=train_iter
        self.learning_rate=learning_rate

        self.image_size = image_size
        self.c_dim = c_dim
        self.batch_size = batch_size

        self.num_landmarks = num_landmarks

        self.save_log_path=save_log_path
        self.save_sample_path=save_sample_path
        self.save_model_path=save_model_path
        self.test_model_path=test_model_path
        self.img_path=img_path

        self.momentum = 0.95
        self.step = 80000  # for lr decay
        self.gamma = 0.1  # for lr decay

        self.weight_initializer = 'xavier'  # random_normal or xavier
        self.weight_initializer_std = 0.01
        self.bias_initializer = 0.0

        self.sigma = 1.5  # sigma for heatmap generation
        self.scale = '1'  # scale for image normalization '255' / '1' / '0'

        self.print_every=1
        self.save_every=5000
        self.sample_every_epoch = False
        self.sample_every=5
        self.sample_grid=9
        self.log_every_epoch=1
        self.log_histograms = True

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        bb_dir = os.path.join(img_path,'Bounding_Boxes')
        self.test_data ='test'  # if mode is TEST, this choose the set to use full/common/challenging/test
        margin = 0.25  # for face crops
        bb_type = 'gt'  # gt/init

        self.debug = False
        self.debug_data_size = 20
        self.compute_nme = True

        self.bb_dictionary = load_bb_dictionary(bb_dir, mode, test_data=self.test_data)

        self.img_menpo_list = load_menpo_image_list(img_path, mode, self.bb_dictionary, image_size, augment=augment,
                                                    margin=margin, bb_type=bb_type, test_data=self.test_data)

        if mode is 'TRAIN':
            train_params = locals()
            print_training_params_to_file(train_params)

    def add_placeholders(self):

        if self.mode == 'TEST':
                self.test_images = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'images')

                self.test_heatmaps_small = tf.placeholder(
                    tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'heatmaps_small')

        elif self.mode == 'TRAIN':
                self.train_images = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'train_images')

                self.train_heatmaps_small = tf.placeholder(
                    tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'train_heatmaps_small')

                if self.compute_nme:
                    self.train_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_lms_small')
                    self.pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'pred_lms_small')


    def heatmaps_network(self, input_images, reuse=None, name='pred_heatmaps'):

        with tf.name_scope(name):

            if self.weight_initializer == 'xavier':
                weight_initializer = contrib.layers.xavier_initializer()
            else:
                weight_initializer = tf.random_normal_initializer(stddev=self.weight_initializer_std)

            bias_init = tf.constant_initializer(self.bias_initializer)

            with tf.variable_scope('heatmaps_network'):
                with tf.name_scope('primary_net'):

                    l1 = conv_relu_pool(input_images, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                        reuse=reuse, var_scope='conv_1')
                    l2 = conv_relu_pool(l1, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                        reuse=reuse, var_scope='conv_2')
                    l3 = conv_relu(l2, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                   reuse=reuse, var_scope='conv_3')

                    l4_1 = conv_relu(l3, 3, 128, conv_dilation=1, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_1')
                    l4_2 = conv_relu(l3, 3, 128, conv_dilation=2, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_2')
                    l4_3 = conv_relu(l3, 3, 128, conv_dilation=3, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_3')
                    l4_4 = conv_relu(l3, 3, 128, conv_dilation=4, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_4')

                    l4 = tf.concat([l4_1, l4_2, l4_3, l4_4], 3, name='conv_4')

                    l5_1 = conv_relu(l4, 3, 256, conv_dilation=1, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_1')
                    l5_2 = conv_relu(l4, 3, 256, conv_dilation=2, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_2')
                    l5_3 = conv_relu(l4, 3, 256, conv_dilation=3, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_3')
                    l5_4 = conv_relu(l4, 3, 256, conv_dilation=4, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_4')

                    l5 = tf.concat([l5_1, l5_2, l5_3, l5_4], 3, name='conv_5')

                    l6 = conv_relu(l5, 1, 512, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_6')
                    l7 = conv_relu(l6, 1, 256, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_7')
                    primary_out = conv(l7, 1, self.num_landmarks, conv_ker_init=weight_initializer,
                                            conv_bias_init=bias_init, reuse=reuse, var_scope='conv_8')

                self.all_layers = [l1, l2, l3, l4, l5, l6, l7, primary_out]

                return primary_out

    def build_model(self):
        if self.mode == 'TEST':
            self.pred_hm_p = self.heatmaps_network(self.test_images)
        elif self.mode == 'TRAIN':
            self.pred_hm_p = self.heatmaps_network(self.train_images,name='pred_heatmaps_train')

    def create_loss_ops(self):

        def l2_loss_norm_eyes(pred_landmarks, real_landmarks, normalize=True, name='NME_loss'):

            with tf.name_scope(name):
                with tf.name_scope('real_pred_landmarks_diff'):
                    landmarks_diff = pred_landmarks - real_landmarks

                if normalize:
                    with tf.name_scope('inter_pupil_dist'):
                        with tf.name_scope('left_eye'):
                            p1 = tf.reduce_mean(tf.slice(real_landmarks, [0, 42, 0], [-1, 6, 2]), axis=1)
                        with tf.name_scope('right_eye'):
                            p2 = tf.reduce_mean(tf.slice(real_landmarks, [0, 36, 0], [-1, 6, 2]), axis=1)
                        eps = 1e-6
                        eye_dist = tf.expand_dims(tf.expand_dims(
                            tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=1)) + eps, axis=1), axis=1)

                    norm_landmarks_diff = landmarks_diff / eye_dist
                    l2_landmarks_norm = tf.reduce_mean(tf.square(norm_landmarks_diff))

                    out = l2_landmarks_norm
                else:
                    l2_landmarks = tf.reduce_mean(tf.square(landmarks_diff))
                    out = l2_landmarks

                return out

        if self.mode is 'TRAIN':
            primary_maps_diff = self.pred_hm_p-self.train_heatmaps_small
            self.total_loss = 1000.*tf.reduce_mean(tf.square(primary_maps_diff))
            # self.total_loss = self.l2_primary

            if self.compute_nme:
                self.nme_loss = l2_loss_norm_eyes(self.pred_lms_small,self.train_lms_small)
            else:
                self.nme_loss = tf.constant(0.)

    def create_summary_ops(self):

            var_summary = [tf.summary.histogram(var.name,var) for var in tf.trainable_variables()]
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summary = [tf.summary.histogram(var.name+'/grads',grad) for grad,var in grads]
            activ_summary = [tf.summary.histogram(layer.name, layer) for layer in self.all_layers]
            l_total = tf.summary.scalar('l_total', self.total_loss)
            l_nme = tf.summary.scalar('l_nme', self.nme_loss)

            if self.log_histograms:
                self.batch_summary_op = tf.summary.merge([l_total, l_nme, var_summary, grad_summary,
                                                          activ_summary])
            else:
                self.batch_summary_op = tf.summary.merge([l_total, l_nme])

    def eval(self):

        self.add_placeholders()
        # build model
        self.build_model()

        num_images = len(self.img_menpo_list)
        img_inds = np.arange(num_images)

        sample_iter = int(1. * num_images / self.sample_grid)

        with tf.Session(config=self.config) as sess:

            # load trained parameters
            print ('loading test model...')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)

            _, model_name = os.path.split(self.test_model_path)

            for i in range(sample_iter):

                batch_inds = img_inds[i * self.sample_grid:(i + 1) * self.sample_grid]

                batch_images, _, batch_maps_gt, _ = \
                    load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                              num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                              save_landmarks=False, primary=True)

                batch_maps_small_pred = sess.run(self.pred_hm_p, {self.test_images: batch_images})

                sample_path_imgs = os.path.join(self.save_sample_path, model_name +'-'+ self.test_data+'-sample-%d-to-%d-1.png' % (
                                        i * self.sample_grid, (i + 1) * self.sample_grid))

                sample_path_maps = os.path.join(self.save_sample_path, model_name +'-'+ self.test_data+ '-sample-%d-to-%d-2.png' % (
                                        i * self.sample_grid, (i + 1) * self.sample_grid))

                sample_path_channels = os.path.join(self.save_sample_path, model_name +'-'+ self.test_data+ '-sample-%d-to-%d-3.png' % (
                    i * self.sample_grid, (i + 1) * self.sample_grid))

                merged_img = merge_images_landmarks_maps(
                    batch_images, batch_maps_small_pred, image_size=self.image_size,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                    scale=self.scale,circle_size=0)

                merged_map = merge_compare_maps(
                    batch_maps_gt, batch_maps_small_pred,image_size=self.image_size/4,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid)

                map_per_channel = map_comapre_channels(
                    batch_images, batch_maps_small_pred,batch_maps_gt, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks, scale=self.scale)

                scipy.misc.imsave(sample_path_imgs, merged_img)
                scipy.misc.imsave(sample_path_maps, merged_map)
                scipy.misc.imsave(sample_path_channels, map_per_channel)

                print ('saved %s' % sample_path_imgs)

    def train(self):
            tf.set_random_seed(1234)
            np.random.seed(1234)
            # build a graph
            # add placeholders
            self.add_placeholders()
            # build model
            self.build_model()
            # create loss ops
            self.create_loss_ops()
            # create summary ops
            self.create_summary_ops()

            # create optimizer and training op
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(self.learning_rate,global_step, self.step, self.gamma, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr,self.momentum)

            train_op = optimizer.minimize(self.total_loss,global_step=global_step)

            with tf.Session(config=self.config) as sess:

                tf.global_variables_initializer().run()

                # create model saver and file writer
                summary_writer = tf.summary.FileWriter(logdir=self.save_log_path, graph=tf.get_default_graph())
                saver = tf.train.Saver()

                print
                print('*** Start Training ***')

                # set random seed
                epoch = 0

                num_train_images = len(self.img_menpo_list)
                if self.debug:
                    num_train_images=self.debug_data_size

                img_inds = np.arange(num_train_images)
                np.random.shuffle(img_inds)

                for step in range(self.train_iter + 1):

                    # get batch images
                    j = step % int(float(num_train_images) / float(self.batch_size))

                    if step > 0 and j == 0:
                        np.random.shuffle(img_inds)  # shuffle data if finished epoch
                        epoch += 1

                    batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

                    batch_images, _, batch_maps_small, batch_lms_small =\
                        load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                                  num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                                  save_landmarks=self.compute_nme, primary=True)

                    feed_dict_train = {self.train_images: batch_images, self.train_heatmaps_small: batch_maps_small}

                    sess.run(train_op, feed_dict_train)

                    # save to log and print status
                    if step == 0 or (step + 1) % self.print_every == 0:

                        if self.compute_nme:
                            batch_maps_small_pred = sess.run(self.pred_hm_p, {self.train_images: batch_images})
                            pred_lms_small = batch_heat_maps_to_image(
                                batch_maps_small_pred, self.batch_size, image_size=self.image_size/4,
                                num_landmarks=self.num_landmarks)

                            feed_dict_log = {
                                self.train_images: batch_images, self.train_heatmaps_small: batch_maps_small,
                                self.train_lms_small: batch_lms_small, self.pred_lms_small: pred_lms_small}
                        else:
                            feed_dict_log = feed_dict_train

                        summary, l_t,l_nme = sess.run([self.batch_summary_op, self.total_loss, self.nme_loss],
                                                      feed_dict_log)

                        summary_writer.add_summary(summary, step)

                        print ('epoch: [%d] step: [%d/%d] primary loss: [%.6f] nme loss: [%.6f] ' % (
                            epoch, step + 1, self.train_iter, l_t, l_nme))

                    # save model
                    if (step + 1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.save_model_path, 'deep_heatmaps'), global_step=step + 1)
                        print ('model/deep-heatmaps-%d saved' % (step + 1))

                    # save images with landmarks
                    if (self.sample_every_epoch is False) and (step == 0 or (step + 1) % self.sample_every == 0):

                            if not self.compute_nme:
                                batch_maps_small_pred = sess.run(self.pred_hm_p,  {self.train_images: batch_images})

                            print 'small map vals', batch_maps_small_pred.min(), batch_maps_small_pred.max()

                            sample_path_imgs = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-1.png'
                                                            % (epoch, step + 1))
                            sample_path_maps = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-2.png'
                                                            % (epoch, step + 1))
                            sample_path_ch_maps = os.path.join(self.save_sample_path, 'epoch-%d-train-iter-%d-3.png'
                                                               % (epoch, step + 1))

                            merged_img = merge_images_landmarks_maps(
                                batch_images, batch_maps_small_pred, image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale,
                                circle_size=0)

                            merged_map = merge_compare_maps(
                                batch_maps_small_pred, batch_maps_small, image_size=self.image_size/4,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid)

                            map_per_channel = map_comapre_channels(batch_images, batch_maps_small_pred, batch_maps_small,
                                                                   image_size=self.image_size/4,
                                                                   num_landmarks=self.num_landmarks,scale=self.scale)

                            scipy.misc.imsave(sample_path_imgs, merged_img)
                            scipy.misc.imsave(sample_path_maps, merged_map)
                            scipy.misc.imsave(sample_path_ch_maps, map_per_channel)

                print('*** Finished Training ***')