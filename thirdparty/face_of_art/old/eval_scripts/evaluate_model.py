from evaluation_functions import *

flags = tf.app.flags

data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
model_path = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/tests/primary/old/no_flip/basic/' \
            'tests_lr_primary_basic_no_flip/0.01/model/deep_heatmaps-80000'

# define paths
flags.DEFINE_string('img_dir', data_dir, 'data directory')
flags.DEFINE_string('test_data', 'test', 'test set to use full/common/challenging/test/art')
flags.DEFINE_string('model_path', model_path, 'model path')

# parameters used to train network
flags.DEFINE_string('network_type', 'Primary', 'network architecture Fusion/Primary')
flags.DEFINE_integer('image_size', 256, 'image size')
flags.DEFINE_integer('c_dim', 3, 'color channels')
flags.DEFINE_integer('num_landmarks', 68, 'number of face landmarks')
flags.DEFINE_integer('scale', 1, 'scale for image normalization 255/1/0')
flags.DEFINE_float('margin', 0.25, 'margin for face crops - % of bb size')
flags.DEFINE_string('bb_type', 'gt', "bb to use -  'gt':for ground truth / 'init':for face detector output")

# choose batch size and debug data size
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_bool('debug', True, 'run in debug mode - use subset of the data')
flags.DEFINE_integer('debug_data_size', 4, 'subset data size to test in debug mode')

# statistics parameters
flags.DEFINE_float('max_error', 0.08, 'error threshold to be considered as failure')
flags.DEFINE_bool('save_log', True, 'save statistics to log_dir')
flags.DEFINE_string('log_path', 'logs/nme_statistics', 'directory for saving NME statistics')

FLAGS = flags.FLAGS


def main(_):

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.log_path):
        tf.gfile.MakeDirs(FLAGS.log_path)

    err = evaluate_heatmap_network(
        model_path=FLAGS.model_path, network_type=FLAGS.network_type, img_path=FLAGS.img_dir,
        test_data=FLAGS.test_data, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size, margin=FLAGS.margin,
        bb_type=FLAGS.bb_type, c_dim=FLAGS.c_dim, scale=FLAGS.scale, num_landmarks=FLAGS.num_landmarks,
        debug=FLAGS.debug, debug_data_size=FLAGS.debug_data_size)

    print_nme_statistics(
        errors=err, model_path=FLAGS.model_path, network_type=FLAGS.network_type, test_data=FLAGS.test_data,
        max_error=FLAGS.max_error, save_log=FLAGS.save_log, log_path=FLAGS.log_path)


if __name__ == '__main__':
    tf.app.run()
