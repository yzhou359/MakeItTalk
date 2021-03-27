from evaluation_functions import *
from glob import glob

flags = tf.app.flags

data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
models_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/ect_like/saved_models/test'

# define paths
flags.DEFINE_string('img_dir', data_dir, 'data directory')
flags.DEFINE_string('test_data', 'test', 'test set to use full/common/challenging/test/art')
flags.DEFINE_string('models_dir', models_dir, 'directory containing multiple models to evaluate and compare')

# parameters used to train network
flags.DEFINE_integer('image_size', 256, 'image size')
flags.DEFINE_integer('c_dim', 3, 'color channels')
flags.DEFINE_integer('num_landmarks', 68, 'number of face landmarks')
flags.DEFINE_integer('scale', 1, 'scale for image normalization 255/1/0')
flags.DEFINE_float('margin', 0.25, 'margin for face crops - % of bb size')
flags.DEFINE_string('bb_type', 'gt', "bb to use -  'gt':for ground truth / 'init':for face detector output")

# choose batch size and debug data size
flags.DEFINE_integer('batch_size', 10, 'batch size')
flags.DEFINE_bool('debug', True, 'run in debug mode - use subset of the data')
flags.DEFINE_integer('debug_data_size', 50, 'subset data size to test in debug mode')

# statistics parameters
flags.DEFINE_float('max_error', 0.08, 'error threshold to be considered as failure')
flags.DEFINE_bool('save_log', True, 'save statistics to log_dir')
flags.DEFINE_string('log_path', 'logs/nme_statistics', 'direcotory for saving NME statistics')

FLAGS = flags.FLAGS


def main(_):

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.log_path):
        tf.gfile.MakeDirs(FLAGS.log_path)

    test_model_dirs = glob(os.path.join(FLAGS.models_dir, '*/'))

    model_names = []
    model_errors = []

    for i, model_dir in enumerate(test_model_dirs):

        model_name = model_dir.split('/')[-2]

        if 'primary' in model_name.lower():
            net_type = 'Primary'
        elif 'fusion' in model_name.lower():
            net_type = 'Fusion'
        else:
            sys.exit('\n*** Error: please give informative names for model directories, including network type! ***')

        model_path = glob(os.path.join(model_dir, '*meta'))[0].split('.meta')[0]

        print ('\n##### EVALUATING MODELS (%d/%d) #####' % (i+1,len(test_model_dirs)))

        tf.reset_default_graph()  # reset graph

        err = evaluate_heatmap_network(
            model_path=model_path, network_type=net_type, img_path=FLAGS.img_dir, test_data=FLAGS.test_data,
            batch_size=FLAGS.batch_size, image_size=FLAGS.image_size, margin=FLAGS.margin,
            bb_type=FLAGS.bb_type, c_dim=FLAGS.c_dim, scale=FLAGS.scale, num_landmarks=FLAGS.num_landmarks,
            debug=FLAGS.debug, debug_data_size=FLAGS.debug_data_size)

        print_nme_statistics(
            errors=err, model_path=model_path, network_type=net_type, test_data=FLAGS.test_data,
            max_error=FLAGS.max_error, save_log=False, log_path=FLAGS.log_path,plot_ced=False)

        model_names.append(model_name)
        model_errors.append(err)

    print_ced_compare_methods(
        method_errors=tuple(model_errors),method_names=tuple(model_names), test_data=FLAGS.test_data,
        log_path=FLAGS.log_path, save_log=FLAGS.save_log)


if __name__ == '__main__':
    tf.app.run()
