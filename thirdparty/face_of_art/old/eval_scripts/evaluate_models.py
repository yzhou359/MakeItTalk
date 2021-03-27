from evaluation_functions import *
from glob import glob

flags = tf.app.flags

data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
models_dir = 'tests_fusion'
pre_train_model_name = 'deep_heatmaps-50000'
datasets=['full','common','challenging','test']

# define paths
flags.DEFINE_string('img_dir', data_dir, 'data directory')
flags.DEFINE_string('models_dir', models_dir, 'directory containing multiple models to evaluate')
flags.DEFINE_string('model_name', pre_train_model_name, "model name. e.g: 'deep_heatmaps-50000'")


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
flags.DEFINE_bool('debug', False, 'run in debug mode - use subset of the data')
flags.DEFINE_integer('debug_data_size', 4, 'subset data size to test in debug mode')

# statistics parameters
flags.DEFINE_float('max_error', 0.08, 'error threshold to be considered as failure')
flags.DEFINE_bool('save_log', True, 'save statistics to log_dir')
flags.DEFINE_string('log_path', 'logs/nme_statistics', 'directory for saving NME statistics')

FLAGS = flags.FLAGS


def main(_):
    model_dirs = glob(os.path.join(FLAGS.models_dir,'*/'))

    for test_data in datasets:
        model_errors=[]
        model_names=[]

        for i, model_dir in enumerate(model_dirs):
            print ('\n##### EVALUATING MODELS ON '+test_data+' set (%d/%d) #####' % (i + 1, len(model_dirs)))
            # create directories if not exist
            log_path = os.path.join(model_dir,'logs/nme_statistics')
            if not os.path.exists(os.path.join(model_dir,'logs')):
                os.mkdir(os.path.join(model_dir,'logs'))
            if not os.path.exists(log_path):
                os.mkdir(log_path)

            model_name = model_dir.split('/')[-2]

            tf.reset_default_graph()  # reset graph

            err = evaluate_heatmap_network(
                model_path=os.path.join(model_dir,'model',FLAGS.model_name), network_type=FLAGS.network_type,
                img_path=FLAGS.img_dir, test_data=test_data, batch_size=FLAGS.batch_size, image_size=FLAGS.image_size,
                margin=FLAGS.margin, bb_type=FLAGS.bb_type, c_dim=FLAGS.c_dim, scale=FLAGS.scale,
                num_landmarks=FLAGS.num_landmarks, debug=FLAGS.debug, debug_data_size=FLAGS.debug_data_size)

            print_nme_statistics(
                errors=err, model_path=os.path.join(model_dir,'model', FLAGS.model_name),
                network_type=FLAGS.network_type, test_data=test_data, max_error=FLAGS.max_error,
                save_log=FLAGS.save_log, log_path=log_path, plot_ced=False)

            model_names.append(model_name)
            model_errors.append(err)

        print_ced_compare_methods(
            method_errors=tuple(model_errors), method_names=tuple(model_names), test_data=test_data,
            log_path=FLAGS.models_dir, save_log=FLAGS.save_log)


if __name__ == '__main__':
    tf.app.run()
