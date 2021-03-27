import scipy.io
import scipy.misc
from glob import glob
import os
import numpy as np
from thirdparty.face_of_art.ops import *
import tensorflow as tf
from tensorflow import contrib
from thirdparty.face_of_art.menpo_functions import *
from thirdparty.face_of_art.logging_functions import *
from thirdparty.face_of_art.data_loading_functions import *


class DeepHeatmapsModel(object):

    """facial landmark localization Network"""

    def __init__(self, mode='TRAIN', train_iter=100000, batch_size=10, learning_rate=1e-3, l_weight_primary=1.,
                 l_weight_fusion=1.,l_weight_upsample=3.,adam_optimizer=True,momentum=0.95,step=100000, gamma=0.1,reg=0,
                 weight_initializer='xavier', weight_initializer_std=0.01, bias_initializer=0.0, image_size=256,c_dim=3,
                 num_landmarks=68, sigma=1.5, scale=1, margin=0.25, bb_type='gt', win_mult=3.33335,
                 augment_basic=True,augment_texture=False, p_texture=0., augment_geom=False, p_geom=0.,
                 output_dir='output', save_model_path='model',
                 save_sample_path='sample', save_log_path='logs', test_model_path='model/deep_heatmaps-50000',
                 pre_train_path='model/deep_heatmaps-50000', load_pretrain=False, load_primary_only=False,
                 img_path='data', test_data='full', valid_data='full', valid_size=0, log_valid_every=5,
                 train_crop_dir='crop_gt_margin_0.25', img_dir_ns='crop_gt_margin_0.25_ns',
                 print_every=100, save_every=5000, sample_every=5000, sample_grid=9, sample_to_log=True,
                 debug_data_size=20, debug=False, epoch_data_dir='epoch_data', use_epoch_data=False, menpo_verbose=True):

        # define some extra parameters

        self.log_histograms = False  # save weight + gradient histogram to log
        self.save_valid_images = True  # sample heat maps of validation images
        self.sample_per_channel = False  # sample heatmaps separately for each landmark

        # for fine-tuning, choose reset_training_op==True. when resuming training, reset_training_op==False
        self.reset_training_op = False

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
        self.load_primary_only = load_primary_only
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
        self.l_weight_primary = l_weight_primary  # primary loss weight
        self.l_weight_fusion = l_weight_fusion  # fusion loss weight
        self.l_weight_upsample = l_weight_upsample  # upsample loss weight

        self.weight_initializer = weight_initializer  # random_normal or xavier
        self.weight_initializer_std = weight_initializer_std
        self.bias_initializer = bias_initializer
        self.adam_optimizer = adam_optimizer

        self.sigma = sigma  # sigma for heatmap generation
        self.scale = scale  # scale for image normalization 255 / 1 / 0
        self.win_mult = win_mult  # gaussian filter size for cpu/gpu approximation: 2 * sigma * win_mult + 1

        self.test_data = test_data  # if mode is TEST, this choose the set to use full/common/challenging/test/art
        self.train_crop_dir = train_crop_dir
        self.img_dir_ns = os.path.join(img_path,img_dir_ns)
        self.augment_basic = augment_basic  # perform basic augmentation (rotation,flip,crop)
        self.augment_texture = augment_texture  # perform artistic texture augmentation (NS)
        self.p_texture = p_texture  # initial probability of artistic texture augmentation
        self.augment_geom = augment_geom  # perform artistic geometric augmentation
        self.p_geom = p_geom  # initial probability of artistic geometric augmentation

        self.valid_size = valid_size
        self.valid_data = valid_data

        # load image, bb and landmark data using menpo
        self.bb_dir = os.path.join(img_path, 'Bounding_Boxes')
        self.bb_dictionary = load_bb_dictionary(self.bb_dir, mode, test_data=self.test_data)

        # use pre-augmented data, to save time during training
        if self.use_epoch_data:
            epoch_0 = os.path.join(self.epoch_data_dir, '0')
            self.img_menpo_list = load_menpo_image_list(
                img_path, train_crop_dir=epoch_0, img_dir_ns=None, mode=mode, bb_dictionary=self.bb_dictionary,
                image_size=self.image_size, test_data=self.test_data, augment_basic=False, augment_texture=False,
                augment_geom=False, verbose=menpo_verbose)
        else:
            self.img_menpo_list = load_menpo_image_list(
                img_path, train_crop_dir, self.img_dir_ns, mode, bb_dictionary=self.bb_dictionary,
                image_size=self.image_size, margin=margin, bb_type=bb_type, test_data=self.test_data,
                augment_basic=augment_basic, augment_texture=augment_texture, p_texture=p_texture,
                augment_geom=augment_geom, p_geom=p_geom, verbose=menpo_verbose)

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

                    self.valid_images_loaded =\
                        np.zeros([self.valid_size, self.image_size, self.image_size, self.c_dim]).astype('float32')
                    self.valid_gt_maps_small_loaded =\
                        np.zeros([self.valid_size, self.image_size / 4, self.image_size / 4,
                                  self.num_landmarks]).astype('float32')
                    self.valid_gt_maps_loaded =\
                        np.zeros([self.valid_size, self.image_size, self.image_size, self.num_landmarks]
                                 ).astype('float32')
                    self.valid_landmarks_loaded = np.zeros([self.valid_size, num_landmarks, 2]).astype('float32')
                    self.valid_landmarks_pred = np.zeros([self.valid_size, self.num_landmarks, 2]).astype('float32')

                    load_images_landmarks_approx_maps_alloc_once(
                        self.valid_img_menpo_list, np.arange(self.valid_size), images=self.valid_images_loaded,
                        maps_small=self.valid_gt_maps_small_loaded, maps=self.valid_gt_maps_loaded,
                        landmarks=self.valid_landmarks_loaded, image_size=self.image_size,
                        num_landmarks=self.num_landmarks, scale=self.scale, win_mult=self.win_mult, sigma=self.sigma,
                        save_landmarks=self.compute_nme)

                    if self.valid_size > self.sample_grid:
                        self.valid_gt_maps_loaded = self.valid_gt_maps_loaded[:self.sample_grid]
                        self.valid_gt_maps_small_loaded = self.valid_gt_maps_small_loaded[:self.sample_grid]
                else:
                    self.val_inds = None

                self.epoch_inds_shuffle = train_val_shuffle_inds_per_epoch(
                    self.val_inds, self.train_inds, train_iter, batch_size, save_log_path)

    def add_placeholders(self):

        if self.mode == 'TEST':
                self.images = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'images')

                self.heatmaps = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, self.num_landmarks], 'heatmaps')

                self.heatmaps_small = tf.placeholder(
                    tf.float32, [None, int(self.image_size/4), int(self.image_size/4), self.num_landmarks], 'heatmaps_small')
                self.lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'lms')
                self.pred_lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'pred_lms')

        elif self.mode == 'TRAIN':
            self.images = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'train_images')

            self.heatmaps = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.num_landmarks], 'train_heatmaps')

            self.heatmaps_small = tf.placeholder(
                tf.float32, [None, int(self.image_size/4), int(self.image_size/4), self.num_landmarks], 'train_heatmaps_small')

            self.train_lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_lms')
            self.train_pred_lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_pred_lms')

            self.valid_lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_lms')
            self.valid_pred_lms = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_pred_lms')

            # self.p_texture_log = tf.placeholder(tf.float32, [])
            # self.p_geom_log = tf.placeholder(tf.float32, [])

            # self.sparse_hm_small = tf.placeholder(tf.float32, [None, int(self.image_size/4), int(self.image_size/4), 1])
            # self.sparse_hm = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])

            if self.sample_to_log:
                row = int(np.sqrt(self.sample_grid))
                self.log_image_map_small = tf.placeholder(
                    tf.uint8, [None, row * int(self.image_size/4), 3 * row * int(self.image_size/4), self.c_dim],
                    'sample_img_map_small')
                self.log_image_map = tf.placeholder(
                    tf.uint8, [None, row * self.image_size, 3 * row * self.image_size, self.c_dim],
                    'sample_img_map')
                if self.sample_per_channel:
                    row = np.ceil(np.sqrt(self.num_landmarks)).astype(np.int64)
                    self.log_map_channels_small = tf.placeholder(
                        tf.uint8, [None, row * int(self.image_size/4), 2 * row * int(self.image_size/4), self.c_dim],
                        'sample_map_channels_small')
                    self.log_map_channels = tf.placeholder(
                        tf.uint8, [None, row * self.image_size, 2 * row * self.image_size, self.c_dim],
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
                    fusion_out = conv(l_fsn_4, 1, self.num_landmarks, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_fsn_5')

                with tf.name_scope('upsample_net'):

                    out = deconv(fusion_out, 8, self.num_landmarks, conv_stride=4,
                                 conv_ker_init=deconv2d_bilinear_upsampling_initializer(
                                     [8, 8, self.num_landmarks, self.num_landmarks]), conv_bias_init=bias_init,
                                 reuse=reuse, var_scope='deconv_1')

                self.all_layers = [l1, l2, l3, l4, l5, l6, l7, primary_out, l_fsn_1, l_fsn_2, l_fsn_3, l_fsn_4,
                                   fusion_out, out]

                return primary_out, fusion_out, out

    def build_model(self):
        self.pred_hm_p, self.pred_hm_f, self.pred_hm_u = self.heatmaps_network(self.images,name='heatmaps_prediction')

    def create_loss_ops(self):

        def nme_norm_eyes(pred_landmarks, real_landmarks, normalize=True, name='NME'):
            """calculate normalized mean error on landmarks - normalize with inter pupil distance"""

            with tf.name_scope(name):
                with tf.name_scope('real_pred_landmarks_rmse'):
                    # calculate RMS ERROR between GT and predicted lms
                    landmarks_rms_err = tf.reduce_mean(
                        tf.sqrt(tf.reduce_sum(tf.square(pred_landmarks - real_landmarks), axis=2)), axis=1)
                if normalize:
                    # normalize RMS ERROR with inter-pupil distance of GT lms
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

            # calculate L2 loss between ideal and predicted heatmaps
            primary_maps_diff = self.pred_hm_p - self.heatmaps_small
            fusion_maps_diff = self.pred_hm_f - self.heatmaps_small
            upsample_maps_diff = self.pred_hm_u - self.heatmaps

            self.l2_primary = tf.reduce_mean(tf.square(primary_maps_diff))
            self.l2_fusion = tf.reduce_mean(tf.square(fusion_maps_diff))
            self.l2_upsample = tf.reduce_mean(tf.square(upsample_maps_diff))

            self.total_loss = 1000.*(self.l_weight_primary * self.l2_primary + self.l_weight_fusion * self.l2_fusion +
                                     self.l_weight_upsample * self.l2_upsample)

            # add weight decay
            self.total_loss += self.reg * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])

            # compute normalized mean error on gt vs. predicted landmarks (for validation)
            if self.compute_nme:
                self.nme_loss = tf.reduce_mean(nme_norm_eyes(self.train_pred_lms, self.train_lms))

            if self.valid_size > 0 and self.compute_nme:
                self.valid_nme_loss = tf.reduce_mean(nme_norm_eyes(self.valid_pred_lms, self.valid_lms))

        elif self.mode == 'TEST' and self.compute_nme:
            self.nme_per_image = nme_norm_eyes(self.pred_lms, self.lms)
            self.nme_loss = tf.reduce_mean(self.nme_per_image)

    def predict_valid_landmarks_in_batches(self, images, session):

        num_images=int(images.shape[0])
        num_batches = int(1.*num_images/self.batch_size)
        if num_batches == 0:
            batch_size = num_images
            num_batches = 1
        else:
            batch_size = self.batch_size

        for j in range(num_batches):

            batch_images = images[j * batch_size:(j + 1) * batch_size,:,:,:]
            batch_maps_pred = session.run(self.pred_hm_u, {self.images: batch_images})
            batch_heat_maps_to_landmarks_alloc_once(
                batch_maps=batch_maps_pred, batch_landmarks=self.valid_landmarks_pred[j * batch_size:(j + 1) * batch_size, :, :],
                batch_size=batch_size,image_size=self.image_size,num_landmarks=self.num_landmarks)

        reminder = num_images-num_batches*batch_size
        if reminder > 0:
            batch_images = images[-reminder:, :, :, :]
            batch_maps_pred = session.run(self.pred_hm_u, {self.images: batch_images})

            batch_heat_maps_to_landmarks_alloc_once(
                batch_maps=batch_maps_pred,
                batch_landmarks=self.valid_landmarks_pred[-reminder:, :, :],
                batch_size=reminder, image_size=self.image_size, num_landmarks=self.num_landmarks)

    def create_summary_ops(self):
        """create summary ops for logging"""

        # loss summary
        l2_primary = tf.summary.scalar('l2_primary', self.l2_primary)
        l2_fusion = tf.summary.scalar('l2_fusion', self.l2_fusion)
        l2_upsample = tf.summary.scalar('l2_upsample', self.l2_upsample)

        l_total = tf.summary.scalar('l_total', self.total_loss)
        self.batch_summary_op = tf.summary.merge([l2_primary,l2_fusion,l2_upsample,l_total])

        if self.compute_nme:
            nme = tf.summary.scalar('nme', self.nme_loss)
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, nme])

        if self.log_histograms:
            var_summary = [tf.summary.histogram(var.name,var) for var in tf.trainable_variables()]
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summary = [tf.summary.histogram(var.name+'/grads',grad) for grad,var in grads]
            activ_summary = [tf.summary.histogram(layer.name, layer) for layer in self.all_layers]
            self.batch_summary_op = tf.summary.merge([self.batch_summary_op, var_summary, grad_summary, activ_summary])

        if self.valid_size > 0 and self.compute_nme:
            self.valid_summary = tf.summary.scalar('valid_nme', self.valid_nme_loss)

        if self.sample_to_log:
            img_map_summary_small = tf.summary.image('compare_map_to_gt_small', self.log_image_map_small)
            img_map_summary = tf.summary.image('compare_map_to_gt', self.log_image_map)

            if self.sample_per_channel:
                map_channels_summary = tf.summary.image('compare_map_channels_to_gt', self.log_map_channels)
                map_channels_summary_small = tf.summary.image('compare_map_channels_to_gt_small',
                                                              self.log_map_channels_small)
                self.img_summary = tf.summary.merge(
                    [img_map_summary, img_map_summary_small,map_channels_summary,map_channels_summary_small])
            else:
                self.img_summary = tf.summary.merge([img_map_summary, img_map_summary_small])

            if self.valid_size >= self.sample_grid:
                img_map_summary_valid_small = tf.summary.image('compare_map_to_gt_small_valid', self.log_image_map_small)
                img_map_summary_valid = tf.summary.image('compare_map_to_gt_valid', self.log_image_map)

                if self.sample_per_channel:
                    map_channels_summary_valid_small = tf.summary.image('compare_map_channels_to_gt_small_valid',
                                                                        self.log_map_channels_small)
                    map_channels_summary_valid = tf.summary.image('compare_map_channels_to_gt_valid',
                                                                  self.log_map_channels)
                    self.img_summary_valid = tf.summary.merge(
                        [img_map_summary_valid,img_map_summary_valid_small,map_channels_summary_valid,
                         map_channels_summary_valid_small])
                else:
                    self.img_summary_valid = tf.summary.merge([img_map_summary_valid, img_map_summary_valid_small])

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

        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()

            # load pre trained weights if load_pretrain==True
            if self.load_pretrain:
                print
                print('*** loading pre-trained weights from: '+self.pre_train_path+' ***')
                if self.load_primary_only:
                    print('*** loading primary-net only ***')
                    primary_var = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                                   ('deconv_' not in v.name) and ('_fsn_' not in v.name)]
                    loader = tf.train.Saver(var_list=primary_var)
                else:
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

            print('\n*** Start Training ***')

            # initialize some variables before training loop
            resume_step = global_step.eval()
            num_train_images = len(self.img_menpo_list)
            batches_in_epoch = int(float(num_train_images) / float(self.batch_size))
            epoch = int(resume_step / batches_in_epoch)
            img_inds = self.epoch_inds_shuffle[epoch, :]
            log_valid = True
            log_valid_images = True

            # allocate space for batch images, maps and landmarks
            batch_images = np.zeros([self.batch_size, self.image_size, self.image_size, self.c_dim]).astype(
                'float32')
            batch_lms = np.zeros([self.batch_size, self.num_landmarks, 2]).astype('float32')
            batch_lms_pred = np.zeros([self.batch_size, self.num_landmarks, 2]).astype('float32')

            batch_maps_small = np.zeros((self.batch_size, int(self.image_size/4),
                                         int(self.image_size/4), self.num_landmarks)).astype('float32')
            batch_maps = np.zeros((self.batch_size, self.image_size, self.image_size,
                                   self.num_landmarks)).astype('float32')

            # create gaussians for heatmap generation
            gaussian_filt_large = create_gaussian_filter(sigma=self.sigma, win_mult=self.win_mult)
            gaussian_filt_small = create_gaussian_filter(sigma=1.*self.sigma/4, win_mult=self.win_mult)

            # training loop
            for step in range(resume_step, self.train_iter):

                j = step % batches_in_epoch  # j==0 if we finished an epoch

                # if we finished an epoch and this isn't the first step
                if step > resume_step and j == 0:
                    epoch += 1
                    img_inds = self.epoch_inds_shuffle[epoch, :]  # get next shuffled image inds
                    log_valid = True
                    log_valid_images = True
                    if self.use_epoch_data:  # if using pre-augmented data, load epoch directory
                        epoch_dir = os.path.join(self.epoch_data_dir, str(epoch))
                        self.img_menpo_list = load_menpo_image_list(
                            self.img_path, train_crop_dir=epoch_dir, img_dir_ns=None, mode=self.mode,
                            bb_dictionary=self.bb_dictionary, image_size=self.image_size, test_data=self.test_data,
                            augment_basic=False, augment_texture=False, augment_geom=False)

                # get batch indices
                batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

                # load batch images, gt maps and landmarks
                load_images_landmarks_approx_maps_alloc_once(
                    self.img_menpo_list, batch_inds, images=batch_images, maps_small=batch_maps_small,
                    maps=batch_maps, landmarks=batch_lms, image_size=self.image_size,
                    num_landmarks=self.num_landmarks, scale=self.scale, gauss_filt_large=gaussian_filt_large,
                    gauss_filt_small=gaussian_filt_small, win_mult=self.win_mult, sigma=self.sigma,
                    save_landmarks=self.compute_nme)

                feed_dict_train = {self.images: batch_images, self.heatmaps: batch_maps,
                                   self.heatmaps_small: batch_maps_small}

                # train on batch
                sess.run(train_op, feed_dict_train)

                # save to log and print status
                if step == resume_step or (step + 1) % self.print_every == 0:

                    # train data log
                    if self.compute_nme:
                        batch_maps_pred = sess.run(self.pred_hm_u, {self.images: batch_images})

                        batch_heat_maps_to_landmarks_alloc_once(
                            batch_maps=batch_maps_pred,batch_landmarks=batch_lms_pred,
                            batch_size=self.batch_size, image_size=self.image_size,
                            num_landmarks=self.num_landmarks)

                        train_feed_dict_log = {
                            self.images: batch_images, self.heatmaps: batch_maps,
                            self.heatmaps_small: batch_maps_small, self.train_lms: batch_lms,
                            self.train_pred_lms: batch_lms_pred}

                        summary, l_p, l_f, l_t, nme = sess.run(
                            [self.batch_summary_op, self.l2_primary, self.l2_fusion, self.total_loss,
                             self.nme_loss],
                            train_feed_dict_log)

                        print (
                            'epoch: [%d] step: [%d/%d] primary loss: [%.6f] fusion loss: [%.6f]'
                            ' total loss: [%.6f] NME: [%.6f]' % (
                                epoch, step + 1, self.train_iter, l_p, l_f, l_t, nme))
                    else:
                        train_feed_dict_log = {self.images: batch_images, self.heatmaps: batch_maps,
                                               self.heatmaps_small: batch_maps_small}

                        summary, l_p, l_f, l_t = sess.run(
                            [self.batch_summary_op, self.l2_primary, self.l2_fusion, self.total_loss],
                            train_feed_dict_log)
                        print (
                            'epoch: [%d] step: [%d/%d] primary loss: [%.6f] fusion loss: [%.6f] total loss: [%.6f]'
                            % (epoch, step + 1, self.train_iter, l_p, l_f, l_t))

                    summary_writer.add_summary(summary, step)

                    # valid data log
                    if self.valid_size > 0 and (log_valid and epoch % self.log_valid_every == 0) \
                            and self.compute_nme:
                        log_valid = False

                        self.predict_valid_landmarks_in_batches(self.valid_images_loaded, sess)
                        valid_feed_dict_log = {
                            self.valid_lms: self.valid_landmarks_loaded,
                            self.valid_pred_lms: self.valid_landmarks_pred}

                        v_summary, v_nme = sess.run([self.valid_summary, self.valid_nme_loss],
                                                      valid_feed_dict_log)
                        summary_writer.add_summary(v_summary, step)
                        print (
                            'epoch: [%d] step: [%d/%d] valid NME: [%.6f]' % (
                                epoch, step + 1, self.train_iter, v_nme))

                # save model
                if (step + 1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.save_model_path, 'deep_heatmaps'), global_step=step + 1)
                    print ('model/deep-heatmaps-%d saved' % (step + 1))

                # save images
                if step == resume_step or (step + 1) % self.sample_every == 0:

                    batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})
                    if not self.compute_nme:
                        batch_maps_pred = sess.run(self.pred_hm_u,  {self.images: batch_images})
                        batch_lms_pred = None

                    merged_img = merge_images_landmarks_maps_gt(
                        batch_images.copy(), batch_maps_pred, batch_maps, landmarks=batch_lms_pred,
                        image_size=self.image_size, num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                        scale=self.scale, circle_size=2, fast=self.fast_img_gen)

                    merged_img_small = merge_images_landmarks_maps_gt(
                        batch_images.copy(), batch_maps_small_pred, batch_maps_small,
                        image_size=self.image_size,
                        num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale,
                        circle_size=0, fast=self.fast_img_gen)

                    if self.sample_per_channel:
                        map_per_channel = map_comapre_channels(
                            batch_images.copy(), batch_maps_pred, batch_maps, image_size=self.image_size,
                            num_landmarks=self.num_landmarks, scale=self.scale)

                        map_per_channel_small = map_comapre_channels(
                            batch_images.copy(), batch_maps_small_pred, batch_maps_small, image_size=int(self.image_size/4),
                            num_landmarks=self.num_landmarks, scale=self.scale)

                    if self.sample_to_log:  # save heatmap images to log
                        if self.sample_per_channel:
                            summary_img = sess.run(
                                self.img_summary, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                   self.log_map_channels: np.expand_dims(map_per_channel, 0),
                                                   self.log_image_map_small: np.expand_dims(merged_img_small, 0),
                                                   self.log_map_channels_small: np.expand_dims(map_per_channel_small, 0)})
                        else:
                            summary_img = sess.run(
                                self.img_summary, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                   self.log_image_map_small: np.expand_dims(merged_img_small, 0)})
                        summary_writer.add_summary(summary_img, step)

                        if (self.valid_size >= self.sample_grid) and self.save_valid_images and\
                                (log_valid_images and epoch % self.log_valid_every == 0):
                            log_valid_images = False

                            batch_maps_small_pred_val,batch_maps_pred_val =\
                                sess.run([self.pred_hm_p,self.pred_hm_u],
                                         {self.images: self.valid_images_loaded[:self.sample_grid]})

                            merged_img_small = merge_images_landmarks_maps_gt(
                                self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                self.valid_gt_maps_small_loaded, image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                                scale=self.scale, circle_size=0, fast=self.fast_img_gen)

                            merged_img = merge_images_landmarks_maps_gt(
                                self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_pred_val,
                                self.valid_gt_maps_loaded, image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                                scale=self.scale, circle_size=2, fast=self.fast_img_gen)

                            if self.sample_per_channel:
                                map_per_channel_small = map_comapre_channels(
                                    self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                    self.valid_gt_maps_small_loaded, image_size=int(self.image_size / 4),
                                    num_landmarks=self.num_landmarks, scale=self.scale)

                                map_per_channel = map_comapre_channels(
                                    self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_pred,
                                    self.valid_gt_maps_loaded, image_size=self.image_size,
                                    num_landmarks=self.num_landmarks, scale=self.scale)

                                summary_img = sess.run(
                                    self.img_summary_valid,
                                    {self.log_image_map: np.expand_dims(merged_img, 0),
                                     self.log_map_channels: np.expand_dims(map_per_channel, 0),
                                     self.log_image_map_small: np.expand_dims(merged_img_small, 0),
                                     self.log_map_channels_small: np.expand_dims(map_per_channel_small, 0)})
                            else:
                                summary_img = sess.run(
                                    self.img_summary_valid,
                                    {self.log_image_map: np.expand_dims(merged_img, 0),
                                     self.log_image_map_small: np.expand_dims(merged_img_small, 0)})

                            summary_writer.add_summary(summary_img, step)
                    else:  # save heatmap images to directory
                        sample_path_imgs = os.path.join(
                            self.save_sample_path, 'epoch-%d-train-iter-%d-1.png' % (epoch, step + 1))
                        sample_path_imgs_small = os.path.join(
                            self.save_sample_path, 'epoch-%d-train-iter-%d-1-s.png' % (epoch, step + 1))
                        scipy.misc.imsave(sample_path_imgs, merged_img)
                        scipy.misc.imsave(sample_path_imgs_small, merged_img_small)

                        if self.sample_per_channel:
                            sample_path_ch_maps = os.path.join(
                                self.save_sample_path, 'epoch-%d-train-iter-%d-3.png' % (epoch, step + 1))
                            sample_path_ch_maps_small = os.path.join(
                                self.save_sample_path, 'epoch-%d-train-iter-%d-3-s.png' % (epoch, step + 1))
                            scipy.misc.imsave(sample_path_ch_maps, map_per_channel)
                            scipy.misc.imsave(sample_path_ch_maps_small, map_per_channel_small)

            print('*** Finished Training ***')

    def get_image_maps(self, test_image, reuse=None, norm=False):
        """ returns heatmaps of input image (menpo image object)"""

        self.add_placeholders()
        # build model
        pred_hm_p, pred_hm_f, pred_hm_u = self.heatmaps_network(self.images, reuse=reuse)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)
            _, model_name = os.path.split(self.test_model_path)

            test_image = test_image.pixels_with_channels_at_back().astype('float32')
            if norm:
                if self.scale is '255':
                    test_image *= 255
                elif self.scale is '0':
                    test_image = 2 * test_image - 1

            map_primary, map_fusion, map_upsample = sess.run(
                [pred_hm_p, pred_hm_f, pred_hm_u], {self.images: np.expand_dims(test_image, 0)})

        return map_primary, map_fusion, map_upsample

    def get_landmark_predictions(self, img_list, pdm_models_dir, clm_model_path, reuse=None, map_to_input_size=False):

        """returns dictionary with landmark predictions of each step of the ECpTp algorithm and ECT"""

        from thirdparty.face_of_art.pdm_clm_functions import feature_based_pdm_corr, clm_correct

        jaw_line_inds = np.arange(0, 17)
        left_brow_inds = np.arange(17, 22)
        right_brow_inds = np.arange(22, 27)

        self.add_placeholders()
        # build model
        _, _, pred_hm_u = self.heatmaps_network(self.images, reuse=reuse)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)
            _, model_name = os.path.split(self.test_model_path)
            e_list = []
            ect_list = []
            ecp_list = []
            ecpt_list = []
            ecptp_jaw_list = []
            ecptp_out_list = []

            for test_image in img_list:

                if map_to_input_size:
                    test_image_transform = test_image[1]
                    test_image=test_image[0]

                # get landmarks for estimation stage
                if test_image.n_channels < 3:
                    test_image_map = sess.run(
                        pred_hm_u, {self.images: np.expand_dims(
                            gray2rgb(test_image.pixels_with_channels_at_back()).astype('float32'), 0)})
                else:
                    test_image_map = sess.run(
                        pred_hm_u, {self.images: np.expand_dims(
                            test_image.pixels_with_channels_at_back().astype('float32'), 0)})
                init_lms = heat_maps_to_landmarks(np.squeeze(test_image_map))

                # get landmarks for part-based correction stage
                p_pdm_lms = feature_based_pdm_corr(lms_init=init_lms, models_dir=pdm_models_dir, train_type='basic')

                # get landmarks for part-based tuning stage
                try:  # clm may not converge
                    pdm_clm_lms = clm_correct(
                        clm_model_path=clm_model_path, image=test_image, map=test_image_map, lms_init=p_pdm_lms)
                except:
                    pdm_clm_lms = p_pdm_lms.copy()

                # get landmarks ECT
                try:  # clm may not converge
                    ect_lms = clm_correct(
                        clm_model_path=clm_model_path, image=test_image, map=test_image_map, lms_init=init_lms)
                except:
                    ect_lms = p_pdm_lms.copy()

                # get landmarks for ECpTp_out (tune jaw and eyebrows)
                ecptp_out = p_pdm_lms.copy()
                ecptp_out[left_brow_inds] = pdm_clm_lms[left_brow_inds]
                ecptp_out[right_brow_inds] = pdm_clm_lms[right_brow_inds]
                ecptp_out[jaw_line_inds] = pdm_clm_lms[jaw_line_inds]

                # get landmarks for ECpTp_jaw (tune jaw)
                ecptp_jaw = p_pdm_lms.copy()
                ecptp_jaw[jaw_line_inds] = pdm_clm_lms[jaw_line_inds]

                if map_to_input_size:
                    ecptp_jaw = test_image_transform.apply(ecptp_jaw)
                    ecptp_out = test_image_transform.apply(ecptp_out)
                    ect_lms = test_image_transform.apply(ect_lms)
                    init_lms = test_image_transform.apply(init_lms)
                    p_pdm_lms = test_image_transform.apply(p_pdm_lms)
                    pdm_clm_lms = test_image_transform.apply(pdm_clm_lms)

                ecptp_jaw_list.append(ecptp_jaw)  # E + p-correction + p-tuning (ECpTp_jaw)
                ecptp_out_list.append(ecptp_out)  # E + p-correction + p-tuning (ECpTp_out)
                ect_list.append(ect_lms)  # ECT prediction
                e_list.append(init_lms)  # init prediction from heatmap network (E)
                ecp_list.append(p_pdm_lms)  # init prediction + part pdm correction (ECp)
                ecpt_list.append(pdm_clm_lms)  # init prediction + part pdm correction + global tuning (ECpT)

            pred_dict = {
                'E': e_list,
                'ECp': ecp_list,
                'ECpT': ecpt_list,
                'ECT': ect_list,
                'ECpTp_jaw': ecptp_jaw_list,
                'ECpTp_out': ecptp_out_list
            }

            return pred_dict
