import tensorflow as tf
from deep_heatmaps_model_primary_fusion import DeepHeatmapsModel
import os

# data_dir ='/mnt/External1/Yarden/deep_face_heatmaps/data/conventional_landmark_detection_dataset/'
data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
pre_train_path = 'saved_models/0.01/model/deep_heatmaps-50000'
output_dir = os.getcwd()

flags = tf.app.flags

flags.DEFINE_string('mode', 'TRAIN', "'TRAIN' or 'TEST'")

# define paths
flags.DEFINE_string('save_model_path', 'model', "directory for saving the model")
flags.DEFINE_string('save_sample_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('save_log_path', 'logs', "directory for saving the log file")
flags.DEFINE_string('img_path', data_dir, "data directory")
flags.DEFINE_string('test_model_path', 'model/deep_heatmaps-5', "saved model to test")
flags.DEFINE_string('test_data','full', 'test set to use full/common/challenging/test/art')

# pretrain parameters
flags.DEFINE_string('pre_train_path', pre_train_path, 'pretrained model path')
flags.DEFINE_bool('load_pretrain', False, "load pretrained weight?")
flags.DEFINE_bool('load_primary_only', True, "load primary weight only?")

flags.DEFINE_integer('image_size', 256, "image size")
flags.DEFINE_integer('c_dim', 3, "color channels")
flags.DEFINE_integer('num_landmarks', 68, "number of face landmarks")

# optimization parameters
flags.DEFINE_integer('train_iter', 100000, 'maximum training iterations')
flags.DEFINE_integer('batch_size', 10, "batch_size")
flags.DEFINE_float('learning_rate', 1e-6, "initial learning rate")
flags.DEFINE_float('momentum', 0.95, 'optimizer momentum')
flags.DEFINE_integer('step', 100000, 'step for lr decay')
flags.DEFINE_float('gamma', 0.1, 'exponential base for lr decay')

# augmentation parameters
flags.DEFINE_bool('augment_basic', True,"use basic augmentation?")
flags.DEFINE_bool('augment_texture', False,"use artistic texture augmentation?")
flags.DEFINE_bool('augment_geom', False,"use artistic geometric augmentation?")
flags.DEFINE_integer('basic_start', 0,  'min epoch to start basic augmentation')
flags.DEFINE_float('p_texture', 0., 'initial probability of artistic texture augmentation')
flags.DEFINE_float('p_geom', 0., 'initial probability of artistic geometric augmentation')
flags.DEFINE_integer('artistic_step', 10, 'increase probability of artistic augmentation every X epochs')
flags.DEFINE_integer('artistic_start', 0, 'min epoch to start artistic augmentation')


# directory of test
flags.DEFINE_string('output_dir', output_dir, "directory for saving test")

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)


def main(_):

    save_model_path = os.path.join(FLAGS.output_dir, FLAGS.save_model_path)
    save_sample_path = os.path.join(FLAGS.output_dir, FLAGS.save_sample_path)
    save_log_path = os.path.join(FLAGS.output_dir, FLAGS.save_log_path)

    # create directories if not exist
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(save_sample_path):
        os.mkdir(save_sample_path)
    if not os.path.exists(save_log_path):
        os.mkdir(save_log_path)

    model = DeepHeatmapsModel(mode=FLAGS.mode, train_iter=FLAGS.train_iter, learning_rate=FLAGS.learning_rate,
                              momentum=FLAGS.momentum, step=FLAGS.step, gamma=FLAGS.gamma, batch_size=FLAGS.batch_size,
                              image_size=FLAGS.image_size, c_dim=FLAGS.c_dim, num_landmarks=FLAGS.num_landmarks,
                              augment_basic=FLAGS.augment_basic, basic_start=FLAGS.basic_start,
                              augment_texture=FLAGS.augment_texture, p_texture=FLAGS.p_texture,
                              augment_geom=FLAGS.augment_geom, p_geom=FLAGS.p_geom,
                              artistic_start=FLAGS.artistic_start, artistic_step=FLAGS.artistic_step,
                              img_path=FLAGS.img_path, save_log_path=save_log_path,
                              save_sample_path=save_sample_path, save_model_path=save_model_path,
                              test_data=FLAGS.test_data, test_model_path=FLAGS.test_model_path,
                              load_pretrain=FLAGS.load_pretrain, load_primary_only=FLAGS.load_primary_only,
                              pre_train_path=FLAGS.pre_train_path)

    if FLAGS.mode == 'TRAIN':
        model.train()
    else:
        model.eval()

if __name__ == '__main__':
    tf.app.run()
