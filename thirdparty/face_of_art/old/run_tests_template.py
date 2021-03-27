import tensorflow as tf
from deep_heatmaps_model_primary_valid import DeepHeatmapsModel
import os
import numpy as np

num_tests = 10
params = np.logspace(-8, -2, num_tests)
max_iter = 80000

output_dir = 'tests_lr_fusion'
data_dir = '../conventional_landmark_detection_dataset'

flags = tf.app.flags
flags.DEFINE_string('output_dir', output_dir, "directory for saving the log file")
flags.DEFINE_string('img_path', data_dir, "data directory")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

for param in params:
    test_name = str(param)
    test_dir = os.path.join(FLAGS.output_dir,test_name)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    print '##### RUNNING TESTS ##### current directory:', test_dir

    save_model_path = os.path.join(test_dir, 'model')
    save_sample_path = os.path.join(test_dir, 'sample')
    save_log_path = os.path.join(test_dir, 'logs')

    # create directories if not exist
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(save_sample_path):
        os.mkdir(save_sample_path)
    if not os.path.exists(save_log_path):
        os.mkdir(save_log_path)

    tf.reset_default_graph()  # reset graph

    model = DeepHeatmapsModel(mode='TRAIN', train_iter=max_iter, learning_rate=param, momentum=0.95, step=80000,
                              gamma=0.1, batch_size=4, image_size=256, c_dim=3, num_landmarks=68,
                              augment_basic=True, basic_start=0, augment_texture=True, p_texture=0.5,
                              augment_geom=True, p_geom=0.5, artistic_start=0, artistic_step=10,
                              img_path=FLAGS.img_path, save_log_path=save_log_path, save_sample_path=save_sample_path,
                              save_model_path=save_model_path)

    model.train()
