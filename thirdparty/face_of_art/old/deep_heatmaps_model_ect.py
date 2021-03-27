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

    def __init__(self, mode='TRAIN', train_iter=500000, learning_rate=0.000001, image_size=256, c_dim=3, batch_size=10,
                 num_landmarks=68, img_path='data', save_log_path='logs', save_sample_path='sample',
                 save_model_path='model',test_model_path='model/deep_heatmaps-1000'):

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
        self.step = 20000  # for lr decay
        self.gamma = 0.05  # for lr decay

        self.weight_initializer = 'random_normal'  # random_normal or xavier
        self.weight_initializer_std = 0.01
        self.bias_initializer = 0.0

        self.l_weight_primary = 100.
        self.l_weight_fusion = 3.*self.l_weight_primary

        self.sigma = 6  # sigma for heatmap generation
        self.scale = 'zero_center'  # scale for image normalization '255' / '1' / 'zero_center'

        self.print_every=2
        self.save_every=100
        self.sample_every_epoch = False
        self.sample_every=10
        self.sample_grid=4
        self.log_every_epoch=1
        self.log_histograms = True

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        bb_dir = '/Users/arik/Desktop/DATA/face_data/300W/Bounding_Boxes/'
        test_data='full'  # if mode is TEST, this choose the set to use full/common/challenging/test
        margin = 0.25  # for face crops
        bb_type = 'gt'  # gt/init

        self.bb_dictionary = load_bb_dictionary(bb_dir, mode, test_data=test_data)

        self.img_menpo_list = load_menpo_image_list(img_path, mode, self.bb_dictionary, image_size,
                                                    margin=margin, bb_type=bb_type, test_data=test_data)

        if mode is 'TRAIN':
            train_params = locals()
            print_training_params_to_file(train_params)

    def add_placeholders(self):

        if self.mode == 'TEST':
                self.test_images = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'images')
                # self.test_landmarks = tf.placeholder(tf.float32, [None, self.num_landmarks * 2], 'landmarks')

                self.test_heatmaps = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.num_landmarks], 'heatmaps')

                self.test_heatmaps_small = tf.placeholder(
                    tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'heatmaps_small')

        elif self.mode == 'TRAIN':
                self.train_images = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'train_images')
                # self.train_landmarks = tf.placeholder(tf.float32, [None, self.num_landmarks*2], 'train_landmarks')

                self.train_heatmaps = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.num_landmarks], 'train_heatmaps')

                self.train_heatmaps_small = tf.placeholder(
                    tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'train_heatmaps_small')

                # self.valid_images = tf.placeholder(
                #     tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'valid_images')
                # # self.valid_landmarks = tf.placeholder(tf.float32, [None, self.num_landmarks * 2], 'valid_landmarks')
                #
                # self.valid_heatmaps = tf.placeholder(
                #     tf.float32, [None, self.image_size, self.image_size, self.num_landmarks], 'valid_heatmaps')
                #
                # self.valid_heatmaps_small = tf.placeholder(
                #     tf.float32,[None, self.image_size / 4, self.image_size / 4, self.num_landmarks], 'valid_heatmaps_small')

    def heatmaps_network(self, input_images, reuse=None, name='pred_heatmaps'):

        with tf.name_scope(name):

            # if training is None:
            #     if self.mode == 'train':
            #         training = True
            #     else:
            #         training = False

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

                with tf.name_scope('fusion_net'):

                    l_fsn_0 = tf.concat([l3, l7], 3, name='conv_3_7_fsn')

                    l_fsn_1_1 = conv_relu(l_fsn_0, 3, 64, conv_dilation=1, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_1_1')
                    l_fsn_1_2 = conv_relu(l_fsn_0, 3, 64, conv_dilation=2, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_1_2')
                    l_fsn_1_3 = conv_relu(l_fsn_0, 3, 64, conv_dilation=3, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_1_3')

                    l_fsn_1 = tf.concat([l_fsn_1_1, l_fsn_1_2, l_fsn_1_3], 3, name='conv_fsn_1')

                    l_fsn_2_1 = conv_relu(l_fsn_1, 3, 64, conv_dilation=1, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_2_1')
                    l_fsn_2_2 = conv_relu(l_fsn_1, 3, 64, conv_dilation=2, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_2_2')
                    l_fsn_2_3 = conv_relu(l_fsn_1, 3, 64, conv_dilation=4, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_2_3')
                    l_fsn_2_4 = conv_relu(l_fsn_1, 5, 64, conv_dilation=3, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_2_4')

                    l_fsn_2 = tf.concat([l_fsn_2_1, l_fsn_2_2, l_fsn_2_3, l_fsn_2_4], 3, name='conv_fsn_2')

                    l_fsn_3_1 = conv_relu(l_fsn_2, 3, 128, conv_dilation=1, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_3_1')
                    l_fsn_3_2 = conv_relu(l_fsn_2, 3, 128, conv_dilation=2, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_3_2')
                    l_fsn_3_3 = conv_relu(l_fsn_2, 3, 128, conv_dilation=4, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_3_3')
                    l_fsn_3_4 = conv_relu(l_fsn_2, 5, 128, conv_dilation=3, conv_ker_init=weight_initializer,
                                          conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_3_4')

                    l_fsn_3 = tf.concat([l_fsn_3_1, l_fsn_3_2, l_fsn_3_3, l_fsn_3_4], 3, name='conv_fsn_3')

                    l_fsn_4 = conv_relu(l_fsn_3, 1, 256, conv_ker_init=weight_initializer,
                                        conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_4')
                    l_fsn_5 = conv(l_fsn_4, 1, self.num_landmarks, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_5')

                with tf.name_scope('upsample_net'):

                    out = deconv(l_fsn_5, 8, self.num_landmarks, conv_stride=4,
                                 conv_ker_init=deconv2d_bilinear_upsampling_initializer(
                                     [8, 8, self.num_landmarks, self.num_landmarks]), conv_bias_init=bias_init,
                                 reuse=reuse, var_scope='deconv_1')

                self.all_layers = [l1, l2, l3, l4, l5, l6, l7, primary_out, l_fsn_1, l_fsn_2, l_fsn_3, l_fsn_4,
                                   l_fsn_5, out]

                return primary_out, out

    def build_model(self):
        if self.mode == 'TEST':
            self.pred_hm_p, self.pred_hm_f = self.heatmaps_network(self.test_images)
        elif self.mode == 'TRAIN':
            self.pred_hm_p,self.pred_hm_f = self.heatmaps_network(self.train_images,name='pred_heatmaps_train')
            # self.pred_landmarks_valid = self.landmarks_network(self.valid_images,name='pred_landmarks_valid')
            # self.pred_landmarks_eval = self.landmarks_network(self.test_images,training=False,reuse=True,name='pred_landmarks_eval')
            # self.pred_landmarks_train = self.landmarks_network(self.train_images, reuse=True, name='pred_landmarks_train')

    def create_loss_ops(self):

        def l2_loss_norm_eyes(pred_landmarks, real_landmarks, normalize=True, name='l2_loss'):

            with tf.name_scope(name):
                with tf.name_scope('real_pred_landmarks_diff'):
                    landmarks_diff = pred_landmarks - real_landmarks

                if normalize:
                    with tf.name_scope('real_landmarks_eye_dist'):
                        with tf.name_scope('left_eye'):
                            p1_out = tf.slice(real_landmarks, [0, 72], [-1, 2])
                            p1_in = tf.slice(real_landmarks, [0, 78], [-1, 2])
                            p1 = (p1_in + p1_out) / 2
                        with tf.name_scope('right_eye'):
                            p2_out = tf.slice(real_landmarks, [0, 90], [-1, 2])
                            p2_in = tf.slice(real_landmarks, [0, 84], [-1, 2])
                            p2 = (p2_in + p2_out) / 2
                        eps = 1e-6
                        eye_dist = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=1)) + eps, axis=1)
                    norm_landmarks_diff = landmarks_diff / eye_dist
                    l2_landmarks_norm = tf.reduce_mean(tf.square(norm_landmarks_diff))

                    out = l2_landmarks_norm
                else:
                    l2_landmarks = tf.reduce_mean(tf.square(landmarks_diff))
                    out = l2_landmarks

                return out

        if self.mode is 'TRAIN':
            primary_maps_diff = self.pred_hm_p-self.train_heatmaps_small
            fusion_maps_diff = self.pred_hm_f - self.train_heatmaps

            self.l2_primary = tf.reduce_mean(tf.square(primary_maps_diff))
            self.l2_fusion = tf.reduce_mean(tf.square(fusion_maps_diff))

            self.total_loss = self.l_weight_primary * self.l2_primary + self.l_weight_fusion * self.l2_fusion

            # self.l2_loss_batch_train = l2_loss_norm_eyes(self.pred_landmarks_train, self.train_landmarks,
            #                                      self.normalize_loss_by_eyes, name='loss_train_batch')
            # with tf.name_scope('losses_not_for_train_step'):
            #     self.l2_loss_train = l2_loss_norm_eyes(self.pred_landmarks_train, self.train_landmarks,
            #                                            self.normalize_loss_by_eyes, name='train')
            #
            #     self.l2_loss_valid = l2_loss_norm_eyes(self.pred_landmarks_valid, self.valid_landmarks,
            #                                            self.normalize_loss_by_eyes, name='valid')
        # else:
            # self.l2_loss_test = l2_loss_norm_eyes(self.pred_landmarks_eval, self.test_landmarks,
            #                                       self.normalize_loss_by_eyes)

    # def predict_landmarks_in_batches(self,image_paths,session):
    #
    #     num_batches = int(1.*len(image_paths)/self.batch_size)
    #     if num_batches == 0:
    #         batch_size = len(image_paths)
    #         num_batches = 1
    #     else:
    #         batch_size = self.batch_size
    #
    #     for i in range(num_batches):
    #         batch_image_paths = image_paths[i * batch_size:(i + 1) * batch_size]
    #         batch_images, _ = \
    #             load_data(batch_image_paths, None, self.image_size, self.num_landmarks, conv=True)
    #         if i == 0:
    #             all_pred_landmarks = session.run(self.pred_landmarks_eval,{self.test_images:batch_images})
    #         else:
    #             batch_pred = session.run(self.pred_landmarks_eval,{self.test_images:batch_images})
    #             all_pred_landmarks = np.concatenate((all_pred_landmarks,batch_pred),0)
    #
    #     reminder = len(image_paths)-num_batches*batch_size
    #     if reminder >0:
    #         reminder_paths = image_paths[-reminder:]
    #         batch_images, _ = \
    #             load_data(reminder_paths, None, self.image_size, self.num_landmarks, conv=True)
    #         batch_pred = session.run(self.pred_landmarks_eval,{self.test_images:batch_images})
    #         all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred), 0)
    #
    #     return all_pred_landmarks

    def create_summary_ops(self):

            var_summary = [tf.summary.histogram(var.name,var) for var in tf.trainable_variables()]
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summary = [tf.summary.histogram(var.name+'/grads',grad) for grad,var in grads]
            activ_summary = [tf.summary.histogram(layer.name, layer) for layer in self.all_layers]
            l2_primary = tf.summary.scalar('l2_primary', self.l2_primary)
            l2_fusion = tf.summary.scalar('l2_fusion', self.l2_fusion)
            l_total = tf.summary.scalar('l_total', self.total_loss)

            if self.log_histograms:
                self.batch_summary_op = tf.summary.merge([l2_primary, l2_fusion, l_total, var_summary, grad_summary,
                                                          activ_summary])
            else:
                self.batch_summary_op = tf.summary.merge([l2_primary, l2_fusion, l_total])

            # l2_train_loss_summary = tf.summary.scalar('l2_loss_train', self.l2_loss_train)
            # l2_valid_loss_summary = tf.summary.scalar('l2_loss_valid', self.l2_loss_valid)
            #
            # self.epoch_summary_op = tf.summary.merge([l2_train_loss_summary, l2_valid_loss_summary])

    def eval(self):

        self.add_placeholders()
        # build model
        self.build_model()

        num_images = len(self.img_menpo_list)
        img_inds = np.arange(num_images)

        sample_iter = int(1. * len(num_images) / self.sample_grid)

        if self.max_test_sample is not None:
            if self.max_test_sample < sample_iter:
                sample_iter = self.max_test_sample

        with tf.Session(config=self.config) as sess:

            # load trained parameters
            print ('loading test model...')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)

            _, model_name = os.path.split(self.test_model_path)

            # if self.new_test_data is False:
            #     # create loss ops
            #     self.create_loss_ops()
            #
            #     all_test_pred_landmarks = self.predict_landmarks_in_batches(test_data_paths, session=sess)
            #     _, all_test_real_landmarks = load_data(None, test_landmarks_paths, self.image_size,
            #                                            self.num_landmarks, conv=True)
            #     all_test_loss = sess.run(self.l2_loss_test, {self.pred_landmarks_eval: all_test_pred_landmarks,
            #                                                  self.test_landmarks: all_test_real_landmarks})
            #     with open(os.path.join(self.save_log_path, model_name+'-test_loss.txt'), 'w') as f:
            #         f.write(str(all_test_loss))

            for i in range(sample_iter):

                batch_inds = img_inds[i * self.sample_grid:(i + 1) * self.sample_grid]

                batch_images, _, _, _ = \
                    load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                              num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                              save_landmarks=False)

                batch_maps_pred, batch_maps_small_pred = sess.run([self.pred_hm_f, self.pred_hm_p],
                                                                  {self.test_images: batch_images})

                sample_path_imgs = os.path.join(self.save_sample_path, model_name + '-sample-%d-to-%d-1.png' % (
                                        i * self.sample_grid, (i + 1) * self.sample_grid))

                sample_path_maps = os.path.join(self.save_sample_path, model_name + '-sample-%d-to-%d-2.png' % (
                                        i * self.sample_grid, (i + 1) * self.sample_grid))

                merged_img = merge_images_landmarks_maps(
                    batch_images, batch_maps_pred, image_size=self.image_size,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale)

                merged_map = merge_compare_maps(
                    batch_maps_small_pred, batch_maps_pred, image_size=self.image_size/4,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid)

                scipy.misc.imsave(sample_path_imgs, merged_img)
                scipy.misc.imsave(sample_path_maps, merged_map)

                print ('saved %s' % sample_path_imgs)

    def train(self):
            tf.set_random_seed(1234)
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
                print_epoch=True

                num_train_images = len(self.img_menpo_list)
                num_train_images=10
                img_inds = np.arange(num_train_images)
                np.random.shuffle(img_inds)

                for step in range(self.train_iter + 1):

                    # get batch images
                    j = step % int(float(num_train_images) / float(self.batch_size))

                    if step > 0 and j == 0:
                        np.random.shuffle(img_inds)  # shuffle data if finished epoch
                        epoch += 1
                        print_epoch=True

                    batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

                    batch_images, batch_maps, batch_maps_small, _ =\
                        load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                                  num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale, save_landmarks=False)

                    feed_dict_train = {self.train_images: batch_images, self.train_heatmaps: batch_maps,
                                       self.train_heatmaps_small: batch_maps_small}

                    sess.run(train_op, feed_dict_train)

                    # print loss every *log_every_epoch* epoch
                    # if step == 0 or (step+1) == self.train_iter or (epoch % self.log_every_epoch ==0 and print_epoch):
                    #     if self.sample_every_epoch is not True:
                    #         print_epoch=False
                    #     all_train_pred_landmarks=self.predict_landmarks_in_batches(train_data_paths,session=sess)
                    #     _,all_train_real_landmarks = load_data(None,train_landmarks_paths,self.image_size,
                    #                                          self.num_landmarks, conv=True)
                    #     all_train_loss = sess.run(self.l2_loss_train,{self.pred_landmarks_train:all_train_pred_landmarks,
                    #                               self.train_landmarks:all_train_real_landmarks})
                    #
                    #     all_valid_pred_landmarks = self.predict_landmarks_in_batches(valid_data_paths,session=sess)
                    #     _, all_valid_real_landmarks = load_data(None, valid_landmarks_paths, self.image_size,
                    #                                             self.num_landmarks, conv=True)
                    #     all_valid_loss = sess.run(self.l2_loss_valid, {self.pred_landmarks_valid: all_valid_pred_landmarks,
                    #                                                    self.valid_landmarks: all_valid_real_landmarks})
                    #     print("--------- EPOCH %d ---------" % (epoch))
                    #     print ('step: [%d/%d] train loss: [%.6f] valid loss: [%.6f]'
                    #            % (step + 1, self.train_iter, all_train_loss, all_valid_loss))
                    #     print("----------------------------")
                    #     summary= sess.run(self.epoch_summary_op,{self.l2_loss_valid:all_valid_loss,self.l2_loss_train:all_train_loss})
                    #     summary_writer.add_summary(summary, epoch)

                    # save to log and print status
                    if step == 0 or (step + 1) % self.print_every == 0:

                        summary, l_p, l_f, l_t = sess.run(
                            [self.batch_summary_op, self.l2_primary,self.l2_fusion,self.total_loss],
                            feed_dict_train)

                        summary_writer.add_summary(summary, step)

                        print ('epoch: [%d] step: [%d/%d] primary loss: [%.6f] fusion loss: [%.6f] total loss: [%.6f]'
                               % (epoch, step + 1, self.train_iter, l_p, l_f, l_t))

                    # save model
                    if (step + 1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.save_model_path, 'deep_heatmaps'), global_step=step + 1)
                        print ('model/deep-heatmaps-%d saved' % (step + 1))

                    # save images with landmarks
                    if self.sample_every_epoch and (epoch % self.log_every_epoch ==0 and print_epoch):
                        print_epoch = False

                        # train_pred = sess.run(self.pred_landmarks_eval, {self.test_images: batch_images})
                        # valid_pred = sess.run(self.pred_landmarks_eval, {self.test_images: valid_images_sample})
                        #
                        # train_sample_path = os.path.join(self.save_sample_path, 'train-epoch-%d.png' % (epoch))
                        # valid_sample_path = os.path.join(self.save_sample_path, 'valid-epoch-%d.png' % (epoch))
                        #
                        # merge_images_train = merge_images_with_landmarks(batch_images, train_pred, self.image_size,
                        #                                                  self.num_landmarks, self.sample_grid)
                        # merge_images_valid = merge_images_with_landmarks(valid_images_sample, valid_pred,
                        #                                                  self.image_size, self.num_landmarks,
                        #                                                  self.sample_grid)
                        #
                        # scipy.misc.imsave(train_sample_path, merge_images_train)
                        # scipy.misc.imsave(valid_sample_path, merge_images_valid)

                    elif (self.sample_every_epoch is False) and (step == 0 or (step + 1) % self.sample_every == 0):

                            batch_maps_pred, batch_maps_small_pred = sess.run([self.pred_hm_f, self.pred_hm_p],
                                                                              {self.train_images: batch_images})

                            print 'map vals', batch_maps_pred.min(), batch_maps_pred.max()
                            print 'small map vals', batch_maps_small_pred.min(), batch_maps_small_pred.max()

                            sample_path_imgs = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-1.png' % (epoch, step + 1))
                            sample_path_maps = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-2.png' % (epoch, step + 1))

                            merged_img = merge_images_landmarks_maps(
                                batch_images, batch_maps_pred, image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale)

                            merged_map = merge_compare_maps(
                                batch_maps_small_pred, batch_maps_pred, image_size=self.image_size/4,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid)

                            scipy.misc.imsave(sample_path_imgs, merged_img)
                            scipy.misc.imsave(sample_path_maps, merged_map)

                print('*** Finished Training ***')
                # evaluate model on test set
                # all_test_pred_landmarks = self.predict_landmarks_in_batches(test_data_paths,session=sess)
                # _, all_test_real_landmarks = load_data(None, test_landmarks_paths, self.image_size,
                #                                        self.num_landmarks, conv=True)
                # all_test_loss = sess.run(self.l2_loss_test, {self.pred_landmarks_test: all_test_pred_landmarks,
                #                                                self.test_landmarks: all_test_real_landmarks})
                #
                # print ('step: [%d/%d] test loss: [%.6f]' % (step, self.train_iter, all_test_loss))
