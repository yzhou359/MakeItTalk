import scipy.io
import scipy.misc
from glob import glob
import os
import numpy as np
from ops import *
import tensorflow as tf
from tensorflow import contrib
from menpo_functions import *
from logging_functions import *
from data_loading_functions import *


class DeepHeatmapsModel(object):

    """facial landmark localization Network"""

    def __init__(self, mode='TRAIN', train_iter=100000, batch_size=10, learning_rate=1e-3, adam_optimizer=True,
                 momentum=0.95, step=100000, gamma=0.1, reg=0, weight_initializer='xavier', weight_initializer_std=0.01,
                 bias_initializer=0.0, image_size=256, c_dim=3, num_landmarks=68, sigma=1.5, scale=1, margin=0.25,
                 bb_type='gt', approx_maps=True, win_mult=3.33335, augment_basic=True, basic_start=0,
                 augment_texture=False, p_texture=0., augment_geom=False, p_geom=0., artistic_step=-1, artistic_start=0,
                 output_dir='output', save_model_path='model', save_sample_path='sample', save_log_path='logs',
                 test_model_path='model/deep_heatmaps-50000', pre_train_path='model/deep_heatmaps-50000',load_pretrain=False,
                 img_path='data', test_data='full', valid_data='full', valid_size=0, log_valid_every=5,
                 train_crop_dir='crop_gt_margin_0.25', img_dir_ns='crop_gt_margin_0.25_ns',
                 print_every=100, save_every=5000, sample_every=5000, sample_grid=9, sample_to_log=True,
                 debug_data_size=20, debug=False, epoch_data_dir='epoch_data', use_epoch_data=False, menpo_verbose=True):

        # define some extra parameters

        self.log_histograms = False  # save weight + gradient histogram to log
        self.save_valid_images = True  # sample heat maps of validation images
        self.log_artistic_augmentation_probs = False  # save p_texture & p_geom to log
        self.sample_per_channel = False  # sample heatmaps separately for each landmark
        self.approx_maps_gpu = False  # create heat-maps on gpu. NOT RECOMMENDED. TODO: REMOVE

        # for fine-tuning, choose reset_training_op==True. when resuming training, reset_training_op==False
        self.reset_training_op = False

        self.allocate_once = True  # create batch images/landmarks/maps zero arrays only once

        self.fast_img_gen = True

        self.compute_nme = True  # compute normalized mean error

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # sampling and logging parameters
        self.print_every = print_every  # print losses to screen + log
        self.save_every = save_every  # save model
        self.sample_every = sample_every  # save images of gen heat maps compared to GT
        self.sample_grid = sample_grid  # number of training images in sample
        self.sample_to_log = sample_to_log  # sample images to log instead of disk
        self.log_valid_every = log_valid_every  # log validation loss (in epochs)

        self.debug = debug
        self.debug_data_size = debug_data_size
        self.use_epoch_data = use_epoch_data
        self.epoch_data_dir = epoch_data_dir

        self.load_pretrain = load_pretrain
        self.pre_train_path = pre_train_path

        self.mode = mode
        self.train_iter = train_iter
        self.learning_rate = learning_rate

        self.image_size = image_size
        self.c_dim = c_dim
        self.batch_size = batch_size

        self.num_landmarks = num_landmarks

        self.save_log_path = save_log_path
        self.save_sample_path = save_sample_path
        self.save_model_path = save_model_path
        self.test_model_path = test_model_path
        self.img_path=img_path

        self.momentum = momentum
        self.step = step  # for lr decay
        self.gamma = gamma  # for lr decay
        self.reg = reg  # weight decay scale

        self.weight_initializer = weight_initializer  # random_normal or xavier
        self.weight_initializer_std = weight_initializer_std
        self.bias_initializer = bias_initializer
        self.adam_optimizer = adam_optimizer

        self.sigma = sigma  # sigma for heatmap generation
        self.scale = scale  # scale for image normalization 255 / 1 / 0
        self.win_mult = win_mult  # gaussian filter size for cpu/gpu approximation: 2 * sigma * win_mult + 1
        self.approx_maps_cpu = approx_maps  # create heat-maps by inserting gaussian filter around landmark locations

        self.test_data = test_data  # if mode is TEST, this choose the set to use full/common/challenging/test/art
        self.train_crop_dir = train_crop_dir
        self.img_dir_ns = os.path.join(img_path, img_dir_ns)
        self.augment_basic = augment_basic  # perform basic augmentation (rotation,flip,crop)
        self.augment_texture = augment_texture  # perform artistic texture augmentation (NS)
        self.p_texture = p_texture  # initial probability of artistic texture augmentation
        self.augment_geom = augment_geom  # perform artistic geometric augmentation
        self.p_geom = p_geom  # initial probability of artistic geometric augmentation
        self.artistic_step = artistic_step  # increase probability of artistic augmentation every X epochs
        self.artistic_start = artistic_start  # min epoch to start artistic augmentation
        self.basic_start = basic_start  # min epoch to start basic augmentation

        self.valid_size = valid_size
        self.valid_data = valid_data

        # load image, bb and landmark data using menpo
        self.bb_dir = os.path.join(img_path, 'Bounding_Boxes')
        self.bb_dictionary = load_bb_dictionary(self.bb_dir, mode, test_data=self.test_data)

        if self.use_epoch_data:
            epoch_0 = os.path.join(self.epoch_data_dir, '0')
            self.img_menpo_list = load_menpo_image_list(
                img_path, train_crop_dir=epoch_0, img_dir_ns=None, mode=mode, bb_dictionary=self.bb_dictionary,
                image_size=self.image_size,test_data=self.test_data, augment_basic=False, augment_texture=False,
                augment_geom=False, verbose=menpo_verbose)
        else:
            self.img_menpo_list = load_menpo_image_list(
                img_path, train_crop_dir, self.img_dir_ns, mode, bb_dictionary=self.bb_dictionary,
                image_size=self.image_size, margin=margin, bb_type=bb_type, test_data=self.test_data,
                augment_basic=(augment_basic and basic_start == 0),
                augment_texture=(augment_texture and artistic_start == 0 and p_texture > 0.), p_texture=p_texture,
                augment_geom=(augment_geom and artistic_start == 0 and p_geom > 0.), p_geom=p_geom,
                verbose=menpo_verbose)

        if mode == 'TRAIN':

            train_params = locals()
            print_training_params_to_file(train_params)  # save init parameters

            self.train_inds = np.arange(len(self.img_menpo_list))

            if self.debug:
                self.train_inds = self.train_inds[:self.debug_data_size]
                self.img_menpo_list = self.img_menpo_list[self.train_inds]

            if valid_size > 0:

                self.valid_bb_dictionary = load_bb_dictionary(self.bb_dir, 'TEST', test_data=self.valid_data)
                self.valid_img_menpo_list = load_menpo_image_list(
                    img_path, train_crop_dir, self.img_dir_ns, 'TEST', bb_dictionary=self.valid_bb_dictionary,
                    image_size=self.image_size, margin=margin, bb_type=bb_type, test_data=self.valid_data,
                    verbose=menpo_verbose)

                np.random.seed(0)
                self.val_inds = np.arange(len(self.valid_img_menpo_list))
                np.random.shuffle(self.val_inds)
                self.val_inds = self.val_inds[:self.valid_size]

                self.valid_img_menpo_list = self.valid_img_menpo_list[self.val_inds]

                if self.approx_maps_cpu:
                    self.valid_images_loaded, self.valid_gt_maps_loaded, self.valid_landmarks_loaded =\
                        load_images_landmarks_approx_maps(
                            self.valid_img_menpo_list, np.arange(self.valid_size), primary=True, image_size=self.image_size,
                            num_landmarks=self.num_landmarks, c_dim=self.c_dim, scale=self.scale, win_mult=self.win_mult,
                            sigma=self.sigma, save_landmarks=True)
                else:
                    self.valid_images_loaded, self.valid_gt_maps_loaded, self.valid_landmarks_loaded =\
                        load_images_landmarks_maps(
                            self.valid_img_menpo_list, np.arange(self.valid_size), primary=True, image_size=self.image_size,
                            c_dim=self.c_dim, num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                            save_landmarks=True)

                if self.allocate_once:
                    self.valid_landmarks_pred = np.zeros([self.valid_size, self.num_landmarks, 2]).astype('float32')

                if self.valid_size > self.sample_grid:
                    self.valid_gt_maps_loaded = self.valid_gt_maps_loaded[:self.sample_grid]
            else:
                self.val_inds = None

            self.epoch_inds_shuffle = train_val_shuffle_inds_per_epoch(
                self.val_inds, self.train_inds, train_iter, batch_size, save_log_path)

    def add_placeholders(self):

        if self.mode == 'TEST':
            self.images = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'images')

            self.heatmaps_small = tf.placeholder(
                tf.float32, [None, int(self.image_size/4), int(self.image_size/4), self.num_landmarks], 'heatmaps_small')
            self.lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'lms_small')
            self.pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'pred_lms_small')

        elif self.mode == 'TRAIN':
            self.images = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'train_images')

            self.heatmaps_small = tf.placeholder(
                tf.float32, [None, int(self.image_size/4), int(self.image_size/4), self.num_landmarks], 'train_heatmaps_small')

            self.train_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_lms_small')
            self.train_pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_pred_lms_small')

            self.valid_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_lms_small')
            self.valid_pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_pred_lms_small')

            self.p_texture_log = tf.placeholder(tf.float32, [])
            self.p_geom_log = tf.placeholder(tf.float32, [])

            self.sparse_hm_small = tf.placeholder(tf.float32, [None, int(self.image_size/4), int(self.image_size/4), 1])

            if self.sample_to_log:
                row = int(np.sqrt(self.sample_grid))
                self.log_image_map = tf.placeholder(
                    tf.uint8, [None,row * int(self.image_size/4), 3 * row *int(self.image_size/4), self.c_dim], 'sample_img_map')
                if self.sample_per_channel:
                    row = np.ceil(np.sqrt(self.num_landmarks)).astype(np.int64)
                    self.log_map_channels = tf.placeholder(
                        tf.uint8, [None, row * int(self.image_size/4), 2 * row * int(self.image_size/4), self.c_dim],
                        'sample_map_channels')

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
        self.pred_hm_p = self.heatmaps_network(self.images,name='heatmaps_prediction')

    def build_hm_generator(self):  # TODO: remove
        # generate heat-maps using:
        # a sparse base (matrix of zeros with 1's in landmark locations) and convolving with a gaussian filter
        print ("*** using convolution to create heat-maps. use this option only with GPU support ***")

        # create gaussian filter
        win_small = int(self.win_mult * self.sigma)
        x_small, y_small = np.mgrid[0:2*win_small+1, 0:2*win_small+1]

        gauss_small = (8. / 3) * self.sigma * gaussian(x_small, y_small, win_small, win_small, sigma=self.sigma)
        gauss_small = tf.constant(gauss_small, tf.float32)
        gauss_small = tf.reshape(gauss_small, [2 * win_small + 1, 2 * win_small + 1, 1, 1])

        # convolve sparse map with gaussian
        self.filt_hm_small = tf.nn.conv2d(self.sparse_hm_small, gauss_small, strides=[1, 1, 1, 1], padding='SAME')
        self.filt_hm_small = tf.transpose(
            tf.concat(tf.split(self.filt_hm_small, self.batch_size, axis=0), 3), [3, 1, 2, 0])

    def create_loss_ops(self):  # TODO: calculate NME on resized maps to 256

        def l2_loss_norm_eyes(pred_landmarks, real_landmarks, normalize=True, name='NME'):

            with tf.name_scope(name):
                with tf.name_scope('real_pred_landmarks_rmse'):
                    landmarks_rms_err = tf.reduce_mean(
                        tf.sqrt(tf.reduce_sum(tf.square(pred_landmarks - real_landmarks), axis=2)), axis=1)
                if normalize:
                    with tf.name_scope('inter_pupil_dist'):
                        with tf.name_scope('left_eye_center'):
                            p1 = tf.reduce_mean(tf.slice(real_landmarks, [0, 42, 0], [-1, 6, 2]), axis=1)
                        with tf.name_scope('right_eye_center'):
                            p2 = tf.reduce_mean(tf.slice(real_landmarks, [0, 36, 0], [-1, 6, 2]), axis=1)

                        eye_dist = tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=1))

                    return landmarks_rms_err / eye_dist
                else:
                    return landmarks_rms_err

        if self.mode is 'TRAIN':
            primary_maps_diff = self.pred_hm_p-self.heatmaps_small
            self.total_loss = 1000.*tf.reduce_mean(tf.square(primary_maps_diff))

            # add weight decay
            self.total_loss += self.reg * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])

            if self.compute_nme:
                self.nme_loss = tf.reduce_mean(l2_loss_norm_eyes(self.train_pred_lms_small,self.train_lms_small))

            if self.valid_size > 0 and self.compute_nme:
                self.valid_nme_loss = tf.reduce_mean(l2_loss_norm_eyes(self.valid_pred_lms_small,self.valid_lms_small))

        elif self.mode == 'TEST' and self.compute_nme:
            self.nme_per_image = l2_loss_norm_eyes(self.pred_lms_small, self.lms_small)
            self.nme_loss = tf.reduce_mean(self.nme_per_image)

    def predict_landmarks_in_batches(self, image_paths, session):

        num_batches = int(1.*len(image_paths)/self.batch_size)
        if num_batches == 0:
            batch_size = len(image_paths)
            num_batches = 1
        else:
            batch_size = self.batch_size

        img_inds = np.arange(len(image_paths))
        for j in range(num_batches):
            batch_inds = img_inds[j * batch_size:(j + 1) * batch_size]

            batch_images, _, batch_lms_small = \
                load_images_landmarks_maps(
                    self.img_menpo_list, batch_inds, primary=True, image_size=self.image_size,
                    c_dim=self.c_dim, num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                    save_landmarks=self.compute_nme)

            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, batch_size=batch_size, image_size=int(self.image_size/4),
                num_landmarks=self.num_landmarks)

            if j == 0:
                all_pred_landmarks = batch_pred_landmarks.copy()
                all_gt_landmarks = batch_lms_small.copy()
            else:
                all_pred_landmarks = np.concatenate((all_pred_landmarks,batch_pred_landmarks),0)
                all_gt_landmarks = np.concatenate((all_gt_landmarks, batch_lms_small), 0)

        reminder = len(image_paths)-num_batches*batch_size

        if reminder > 0:
            reminder_inds = img_inds[-reminder:]

            batch_images, _, batch_lms_small = \
                load_images_landmarks_maps(
                    self.img_menpo_list, reminder_inds, primary=True, image_size=self.image_size,
                    c_dim=self.c_dim, num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                    save_landmarks=self.compute_nme)

            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, batch_size=reminder, image_size=int(self.image_size/4),
                num_landmarks=self.num_landmarks)

            all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred_landmarks), 0)
            all_gt_landmarks = np.concatenate((all_gt_landmarks, batch_lms_small), 0)

        return all_pred_landmarks, all_gt_landmarks

    def predict_landmarks_in_batches_loaded(self, images, session):

        num_images = int(images.shape[0])
        num_batches = int(1.*num_images/self.batch_size)
        if num_batches == 0:
            batch_size = num_images
            num_batches = 1
        else:
            batch_size = self.batch_size

        for j in range(num_batches):

            batch_images = images[j * batch_size:(j + 1) * batch_size,:,:,:]
            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            if self.allocate_once:
                batch_heat_maps_to_landmarks_alloc_once(
                    batch_maps=batch_maps_small_pred,
                    batch_landmarks=self.valid_landmarks_pred[j * batch_size:(j + 1) * batch_size, :, :],
                    batch_size=batch_size, image_size=int(self.image_size/4), num_landmarks=self.num_landmarks)
            else:
                batch_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, batch_size=batch_size, image_size=int(self.image_size/4),
                    num_landmarks=self.num_landmarks)

                if j == 0:
                    all_pred_landmarks = batch_pred_landmarks.copy()
                else:
                    all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred_landmarks), 0)

        reminder = num_images-num_batches*batch_size
        if reminder > 0:

            batch_images = images[-reminder:, :, :, :]
            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            if self.allocate_once:
                batch_heat_maps_to_landmarks_alloc_once(
                    batch_maps=batch_maps_small_pred,
                    batch_landmarks=self.valid_landmarks_pred[-reminder:, :, :],
                    batch_size=reminder, image_size=int(self.image_size/4), num_landmarks=self.num_landmarks)
            else:
                batch_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, batch_size=reminder, image_size=int(self.image_size/4),
                    num_landmarks=self.num_landmarks)

                all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred_landmarks), 0)

        if not self.allocate_once:
            return all_pred_landmarks

    def create_summary_ops(self):

        self.batch_summary_op = tf.summary.scalar('l_total', self.total_loss)

        if self.compute_nme:
            l_nme = tf.summary.scalar('l_nme', self.nme_loss)
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, l_nme])

        if self.log_histograms:
            var_summary = [tf.summary.histogram(var.name, var) for var in tf.trainable_variables()]
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summary = [tf.summary.histogram(var.name + '/grads', grad) for grad, var in grads]
            activ_summary = [tf.summary.histogram(layer.name, layer) for layer in self.all_layers]
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, var_summary, grad_summary, activ_summary])

        if self.augment_texture and self.log_artistic_augmentation_probs:
            p_texture_summary = tf.summary.scalar('p_texture', self.p_texture_log)
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, p_texture_summary])

        if self.augment_geom and self.log_artistic_augmentation_probs:
            p_geom_summary = tf.summary.scalar('p_geom', self.p_geom_log)
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, p_geom_summary])

        if self.valid_size > 0 and self.compute_nme:
            self.valid_summary = tf.summary.scalar('valid_l_nme', self.valid_nme_loss)

        if self.sample_to_log:
            img_map_summary =tf.summary.image('compare_map_to_gt',self.log_image_map)
            if self.sample_per_channel:
                map_channels_summary = tf.summary.image('compare_map_channels_to_gt', self.log_map_channels)
                self.img_summary = tf.summary.merge([img_map_summary, map_channels_summary])
            else:
                self.img_summary = img_map_summary
            if self.valid_size >= self.sample_grid:
                img_map_summary_valid = tf.summary.image('compare_map_to_gt_valid', self.log_image_map)
                if self.sample_per_channel:
                    map_channels_summary_valid = tf.summary.image('compare_map_channels_to_gt_valid', self.log_map_channels)
                    self.img_summary_valid = tf.summary.merge([img_map_summary_valid, map_channels_summary_valid])
                else:
                    self.img_summary_valid = img_map_summary_valid

    def eval(self):

        self.add_placeholders()
        # build model
        self.build_model()
        self.create_loss_ops()

        if self.debug:
            self.img_menpo_list = self.img_menpo_list[:np.min([self.debug_data_size, len(self.img_menpo_list)])]

        num_images = len(self.img_menpo_list)
        img_inds = np.arange(num_images)

        sample_iter = np.ceil(1. * num_images / self.sample_grid).astype('int')

        with tf.Session(config=self.config) as sess:

            # load trained parameters
            print ('loading test model...')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)

            _, model_name = os.path.split(self.test_model_path)

            gt_provided = self.img_menpo_list[0].has_landmarks  # check if GT landmarks provided

            for i in range(sample_iter):

                batch_inds = img_inds[i * self.sample_grid:(i + 1) * self.sample_grid]

                if not gt_provided:
                    batch_images = load_images(self.img_menpo_list, batch_inds, image_size=self.image_size,
                                               c_dim=self.c_dim, scale=self.scale)

                    batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})

                    batch_maps_gt = None
                else:
                    # TODO: add option for approx maps + allocate once
                    batch_images, batch_maps_gt, _ = \
                        load_images_landmarks_maps(
                            self.img_menpo_list, batch_inds, primary=True, image_size=self.image_size,
                            c_dim=self.c_dim, num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                            save_landmarks=False)

                    batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})

                sample_path_imgs = os.path.join(
                    self.save_sample_path, model_name +'-'+ self.test_data+'-sample-%d-to-%d-1.png' % (
                        i * self.sample_grid, (i + 1) * self.sample_grid))

                merged_img = merge_images_landmarks_maps_gt(
                    batch_images.copy(), batch_maps_small_pred, batch_maps_gt, image_size=self.image_size,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale, circle_size=0,
                    fast=self.fast_img_gen)

                scipy.misc.imsave(sample_path_imgs, merged_img)

                if self.sample_per_channel:
                    map_per_channel = map_comapre_channels(
                        batch_images.copy(), batch_maps_small_pred,batch_maps_gt, image_size=int(self.image_size/4),
                        num_landmarks=self.num_landmarks, scale=self.scale)

                    sample_path_channels = os.path.join(
                        self.save_sample_path, model_name + '-' + self.test_data + '-sample-%d-to-%d-3.png' % (
                            i * self.sample_grid, (i + 1) * self.sample_grid))

                    scipy.misc.imsave(sample_path_channels, map_per_channel)

                print ('saved %s' % sample_path_imgs)

            if self.compute_nme and self.test_data in ['full', 'challenging', 'common', 'training', 'test']:
                print ('\n Calculating NME on: ' + self.test_data + '...')
                pred_lms, lms_gt = self.predict_landmarks_in_batches(self.img_menpo_list, sess)
                nme = sess.run(self.nme_loss, {self.pred_lms_small: pred_lms, self.lms_small: lms_gt})
                print ('NME on ' + self.test_data + ': ' + str(nme))

    def train(self):
        # set random seed
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
        if self.adam_optimizer:
            optimizer = tf.train.AdamOptimizer(lr)
        else:
            optimizer = tf.train.MomentumOptimizer(lr, self.momentum)

        train_op = optimizer.minimize(self.total_loss,global_step=global_step)

        # TODO: remove
        if self.approx_maps_gpu:  # create heat-maps using tf convolution. use only with GPU support!
            self.build_hm_generator()

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()

            # load pre trained weights if load_pretrain==True
            if self.load_pretrain:
                print
                print('*** loading pre-trained weights from: '+self.pre_train_path+' ***')
                loader = tf.train.Saver()
                loader.restore(sess, self.pre_train_path)
                print("*** Model restore finished, current global step: %d" % global_step.eval())

            # for fine-tuning, choose reset_training_op==True. when resuming training, reset_training_op==False
            if self.reset_training_op:
                print ("resetting optimizer and global step")
                opt_var_list = [optimizer.get_slot(var, name) for name in optimizer.get_slot_names()
                                 for var in tf.global_variables() if optimizer.get_slot(var, name) is not None]
                opt_var_list_init = tf.variables_initializer(opt_var_list)
                opt_var_list_init.run()
                sess.run(global_step.initializer)

            # create model saver and file writer
            summary_writer = tf.summary.FileWriter(logdir=self.save_log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print
            print('*** Start Training ***')

            # initialize some variables before training loop
            resume_step = global_step.eval()
            num_train_images = len(self.img_menpo_list)
            batches_in_epoch = int(float(num_train_images) / float(self.batch_size))
            epoch = int(resume_step / batches_in_epoch)
            img_inds = self.epoch_inds_shuffle[epoch, :]
            p_texture = self.p_texture
            p_geom = self.p_geom
            artistic_reload = False
            basic_reload = True
            log_valid = True
            log_valid_images = True

            if self.allocate_once:
                batch_images = np.zeros([self.batch_size, self.image_size, self.image_size, self.c_dim]).astype('float32')
                batch_lms_small = np.zeros([self.batch_size, self.num_landmarks, 2]).astype('float32')
                batch_lms_small_pred = np.zeros([self.batch_size, self.num_landmarks, 2]).astype('float32')
                if self.approx_maps_gpu:
                    batch_hm_base_small = np.zeros((self.batch_size * self.num_landmarks,
                                                    int(self.image_size/4), int(self.image_size/4), 1)).astype('float32')
                else:
                    batch_maps_small = np.zeros((self.batch_size, int(self.image_size/4),
                                                 int(self.image_size/4), self.num_landmarks)).astype('float32')

            if self.approx_maps_cpu:
                gaussian_filt = create_gaussian_filter(sigma=self.sigma, win_mult=self.win_mult)

            for step in range(resume_step, self.train_iter):

                j = step % batches_in_epoch  # j==0 if we finished an epoch

                if step > resume_step and j == 0:  # if we finished an epoch and this isn't the first step
                    epoch += 1
                    img_inds = self.epoch_inds_shuffle[epoch, :]  # get next shuffled image inds
                    artistic_reload = True
                    log_valid = True
                    log_valid_images = True
                    if self.use_epoch_data:
                        epoch_dir = os.path.join(self.epoch_data_dir, str(epoch))
                        self.img_menpo_list = load_menpo_image_list(
                            self.img_path, train_crop_dir=epoch_dir, img_dir_ns=None, mode=self.mode,
                            bb_dictionary=self.bb_dictionary, image_size=self.image_size, test_data=self.test_data,
                            augment_basic=False, augment_texture=False, augment_geom=False)

                # add basic augmentation (if basic_start > 0 and augment_basic is True)
                if basic_reload and (epoch >= self.basic_start) and self.basic_start > 0 and self.augment_basic:
                    basic_reload = False
                    self.img_menpo_list = reload_menpo_image_list(
                        self.img_path, self.train_crop_dir, self.img_dir_ns, self.mode, self.train_inds,
                        image_size=self.image_size, augment_basic=self.augment_basic,
                        augment_texture=(self.augment_texture and epoch >= self.artistic_start), p_texture=p_texture,
                        augment_geom=(self.augment_geom and epoch >= self.artistic_start), p_geom=p_geom)
                    print ("****** adding basic augmentation ******")

                # increase artistic augmentation probability
                if ((epoch % self.artistic_step == 0 and epoch >= self.artistic_start and self.artistic_step != -1)
                    or (epoch == self.artistic_start)) and (self.augment_geom or self.augment_texture)\
                        and artistic_reload:

                    artistic_reload = False

                    if epoch == self.artistic_start:
                        print ("****** adding artistic augmentation ******")
                        print ("****** augment_geom: " + str(self.augment_geom) + ", p_geom: " + str(p_geom) + " ******")
                        print ("****** augment_texture: " + str(self.augment_texture) + ", p_texture: " +
                               str(p_texture) + " ******")

                    if epoch % self.artistic_step == 0 and self.artistic_step != -1:
                        print ("****** increasing artistic augmentation probability ******")

                        p_geom = 1.- 0.95 ** (epoch/self.artistic_step)
                        p_texture = 1. - 0.95 ** (epoch/self.artistic_step)

                        print ("****** augment_geom: " + str(self.augment_geom) + ", p_geom: " + str(p_geom) + " ******")
                        print ("****** augment_texture: " + str(self.augment_texture) + ", p_texture: " +
                               str(p_texture) + " ******")

                    self.img_menpo_list = reload_menpo_image_list(
                        self.img_path, self.train_crop_dir, self.img_dir_ns, self.mode, self.train_inds,
                        image_size=self.image_size, augment_basic=(self.augment_basic and epoch >= self.basic_start),
                        augment_texture=self.augment_texture, p_texture=p_texture,
                        augment_geom=self.augment_geom, p_geom=p_geom)

                # get batch images
                batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

                if self.approx_maps_gpu:  # TODO: remove
                    if self.allocate_once:
                        load_images_landmarks_alloc_once(
                            self.img_menpo_list, batch_inds, images=batch_images, landmarks_small=batch_lms_small,
                            landmarks=None, primary=True, image_size=self.image_size, scale=self.scale)

                        create_heat_maps_base_alloc_once(
                            landmarks_small=batch_lms_small.astype(int), landmarks=None,
                            hm_small=batch_hm_base_small, hm_large=None, primary=True, num_images=self.batch_size,
                            num_landmarks=self.num_landmarks)
                    else:
                        batch_images, batch_lms_small = load_images_landmarks(
                            self.img_menpo_list, batch_inds, primary=True, image_size=self.image_size, c_dim=self.c_dim,
                            num_landmarks=self.num_landmarks, scale=self.scale)

                        batch_hm_base_small = create_heat_maps_base(
                            landmarks_small=batch_lms_small.astype(int), landmarks=None, primary=True,
                            num_images=self.batch_size, image_size=self.image_size, num_landmarks=self.num_landmarks)

                    batch_maps_small = sess.run(self.filt_hm_small, {self.sparse_hm_small: batch_hm_base_small})
                elif self.approx_maps_cpu:
                    if self.allocate_once:
                        load_images_landmarks_approx_maps_alloc_once(
                            self.img_menpo_list, batch_inds, images=batch_images, maps_small=batch_maps_small,
                            maps=None, landmarks=batch_lms_small, primary=True, image_size=self.image_size,
                            num_landmarks=self.num_landmarks, scale=self.scale, gauss_filt_small=gaussian_filt,
                            win_mult=self.win_mult, sigma=self.sigma, save_landmarks=self.compute_nme)
                    else:
                        batch_images, batch_maps_small, batch_lms_small = load_images_landmarks_approx_maps(
                            self.img_menpo_list, batch_inds, primary=True, image_size=self.image_size,
                            num_landmarks=self.num_landmarks, c_dim=self.c_dim, scale=self.scale,
                            gauss_filt_small=gaussian_filt, win_mult=self.win_mult, sigma=self.sigma,
                            save_landmarks=self.compute_nme)
                else:
                    if self.allocate_once:
                        load_images_landmarks_maps_alloc_once(
                            self.img_menpo_list, batch_inds, images=batch_images, maps_small=batch_maps_small,
                            landmarks=batch_lms_small, maps=None, primary=True, image_size=self.image_size,
                            num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                            save_landmarks=self.compute_nme)
                    else:
                        batch_images, batch_maps_small, batch_lms_small = load_images_landmarks_maps(
                            self.img_menpo_list, batch_inds, primary=True, image_size=self.image_size, c_dim=self.c_dim,
                            num_landmarks=self.num_landmarks, scale=self.scale, sigma=self.sigma,
                            save_landmarks=self.compute_nme)

                feed_dict_train = {self.images: batch_images, self.heatmaps_small: batch_maps_small}

                sess.run(train_op, feed_dict_train)

                # save to log and print status
                if step == resume_step or (step + 1) % self.print_every == 0:

                    # log probability of artistic augmentation
                    if self.log_artistic_augmentation_probs and (self.augment_geom or self.augment_texture):
                        if self.augment_geom and not self.augment_texture:
                            art_augment_prob_dict = {self.p_geom_log: p_geom}
                        elif self.augment_texture and not self.augment_geom:
                            art_augment_prob_dict = {self.p_texture_log: p_texture}
                        else:
                            art_augment_prob_dict = {self.p_texture_log: p_texture, self.p_geom_log: p_geom}

                    # train data log
                    if self.compute_nme:
                        batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})
                        if self.allocate_once:
                            batch_heat_maps_to_landmarks_alloc_once(
                                batch_maps=batch_maps_small_pred, batch_landmarks=batch_lms_small_pred,
                                batch_size=self.batch_size, image_size=int(self.image_size/4),
                                num_landmarks=self.num_landmarks)
                        else:
                            batch_lms_small_pred = batch_heat_maps_to_landmarks(
                                batch_maps_small_pred, self.batch_size, image_size=int(self.image_size/4),
                                num_landmarks=self.num_landmarks)

                        train_feed_dict_log = {
                            self.images: batch_images, self.heatmaps_small: batch_maps_small,
                            self.train_lms_small: batch_lms_small, self.train_pred_lms_small: batch_lms_small_pred}
                        if self.log_artistic_augmentation_probs and (self.augment_geom or self.augment_texture):
                            train_feed_dict_log.update(art_augment_prob_dict)

                        summary, l_t, l_nme = sess.run(
                            [self.batch_summary_op, self.total_loss, self.nme_loss], train_feed_dict_log)

                        print (
                            'epoch: [%d] step: [%d/%d] primary loss: [%.6f] NME: [%.6f]' % (
                                epoch, step + 1, self.train_iter, l_t, l_nme))
                    else:
                        train_feed_dict_log = {self.images: batch_images, self.heatmaps_small: batch_maps_small}
                        if self.log_artistic_augmentation_probs and (self.augment_geom or self.augment_texture):
                            train_feed_dict_log.update(art_augment_prob_dict)

                        summary, l_t = sess.run(
                            [self.batch_summary_op, self.total_loss], train_feed_dict_log)

                        print (
                            'epoch: [%d] step: [%d/%d] primary loss: [%.6f]' % (
                                epoch, step + 1, self.train_iter, l_t))

                    summary_writer.add_summary(summary, step)

                    # valid data log
                    if self.valid_size > 0 and (log_valid and epoch % self.log_valid_every == 0)\
                            and self.compute_nme:
                        log_valid = False

                        if self.allocate_once:
                            self.predict_landmarks_in_batches_loaded(self.valid_images_loaded, sess)
                            valid_feed_dict_log = {
                                self.valid_lms_small: self.valid_landmarks_loaded,
                                self.valid_pred_lms_small: self.valid_landmarks_pred}
                        else:
                            valid_pred_lms = self.predict_landmarks_in_batches_loaded(self.valid_images_loaded, sess)
                            valid_feed_dict_log = {
                                self.valid_lms_small: self.valid_landmarks_loaded,
                                self.valid_pred_lms_small: valid_pred_lms}

                        v_summary,l_v_nme = sess.run([self.valid_summary, self.valid_nme_loss], valid_feed_dict_log)
                        summary_writer.add_summary(v_summary, step)

                        print (
                            'epoch: [%d] step: [%d/%d] valid NME: [%.6f]' % (
                                epoch, step + 1, self.train_iter, l_v_nme))

                # save model
                if (step + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.save_model_path, 'deep_heatmaps'), global_step=step + 1)
                    print ('model/deep-heatmaps-%d saved' % (step + 1))

                # save images. TODO: add option to allocate once
                if step == resume_step or (step + 1) % self.sample_every == 0:

                    if not self.compute_nme:
                        batch_maps_small_pred = sess.run(self.pred_hm_p,  {self.images: batch_images})
                        batch_lms_small_pred=None

                    merged_img = merge_images_landmarks_maps_gt(
                        batch_images.copy(), batch_maps_small_pred, batch_maps_small,
                        landmarks=batch_lms_small_pred, image_size=self.image_size,
                        num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale,
                        circle_size=0, fast=self.fast_img_gen)

                    if self.sample_per_channel:
                        map_per_channel = map_comapre_channels(
                            batch_images.copy(), batch_maps_small_pred,batch_maps_small,
                            image_size=int(self.image_size/4), num_landmarks=self.num_landmarks, scale=self.scale)

                    if self.sample_to_log:
                        if self.sample_per_channel:
                            summary_img = sess.run(
                                self.img_summary, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                   self.log_map_channels: np.expand_dims(map_per_channel, 0)})
                        else:
                            summary_img = sess.run(
                                self.img_summary, {self.log_image_map: np.expand_dims(merged_img, 0)})

                        summary_writer.add_summary(summary_img, step)

                        if (self.valid_size >= self.sample_grid) and self.save_valid_images and\
                                (log_valid_images and epoch % self.log_valid_every == 0):
                            log_valid_images=False

                            batch_maps_small_pred_val = sess.run(
                                self.pred_hm_p, {self.images: self.valid_images_loaded[:self.sample_grid]})

                            merged_img = merge_images_landmarks_maps_gt(
                                self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                self.valid_gt_maps_loaded, image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                                scale=self.scale, circle_size=0, fast=self.fast_img_gen)

                            if self.sample_per_channel:
                                map_per_channel = map_comapre_channels(
                                    self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                    self.valid_gt_maps_loaded, image_size=int(self.image_size/4),
                                    num_landmarks=self.num_landmarks, scale=self.scale)

                                summary_img = sess.run(
                                    self.img_summary_valid, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                       self.log_map_channels: np.expand_dims(map_per_channel, 0)})
                            else:
                                summary_img = sess.run(
                                    self.img_summary_valid, {self.log_image_map: np.expand_dims(merged_img, 0)})
                            summary_writer.add_summary(summary_img, step)

                    else:
                        sample_path_imgs = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-1.png'
                                                        % (epoch, step + 1))
                        scipy.misc.imsave(sample_path_imgs, merged_img)
                        if self.sample_per_channel:
                            sample_path_ch_maps = os.path.join(self.save_sample_path, 'epoch-%d-train-iter-%d-3.png'
                                                               % (epoch, step + 1))
                            scipy.misc.imsave(sample_path_ch_maps, map_per_channel)

            print('*** Finished Training ***')

    def get_maps_image(self, test_image, reuse=None):
        self.add_placeholders()
        # build model
        pred_hm_p = self.heatmaps_network(self.images,reuse=reuse)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)
            _, model_name = os.path.split(self.test_model_path)

            test_image = test_image.pixels_with_channels_at_back().astype('float32')
            if self.scale is '255':
                test_image *= 255
            elif self.scale is '0':
                test_image = 2 * test_image - 1

            test_image_map = sess.run(pred_hm_p, {self.images: np.expand_dims(test_image,0)})

        return test_image_map
