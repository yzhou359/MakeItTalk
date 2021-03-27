import tensorflow as tf
from deep_heatmaps_model_primary_valid import DeepHeatmapsModel

# data_dir ='/mnt/External1/Yarden/deep_face_heatmaps/data/conventional_landmark_detection_dataset/'
data_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
pre_train_path = 'saved_models/0.01/model/deep_heatmaps-50000'

flags = tf.app.flags
flags.DEFINE_string('mode', 'TRAIN', "'TRAIN' or 'TEST'")
flags.DEFINE_string('save_model_path', 'model', "directory for saving the model")
flags.DEFINE_string('save_sample_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('save_log_path', 'logs', "directory for saving the log file")
flags.DEFINE_string('img_path', data_dir, "data directory")
flags.DEFINE_string('test_model_path', 'model/deep_heatmaps-5', 'saved model to test')
flags.DEFINE_string('test_data', 'full', 'dataset to test: full/common/challenging/test/art')
flags.DEFINE_string('pre_train_path', pre_train_path, 'pretrained model path')

FLAGS = flags.FLAGS


def main(_):

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.save_model_path):
        tf.gfile.MakeDirs(FLAGS.save_model_path)
    if not tf.gfile.Exists(FLAGS.save_sample_path):
        tf.gfile.MakeDirs(FLAGS.save_sample_path)
    if not tf.gfile.Exists(FLAGS.save_log_path):
        tf.gfile.MakeDirs(FLAGS.save_log_path)

    model = DeepHeatmapsModel(mode=FLAGS.mode, train_iter=80000, learning_rate=1e-11, momentum=0.95, step=80000,
                              gamma=0.1, batch_size=4, image_size=256, c_dim=3, num_landmarks=68,
                              augment_basic=True, basic_start=1, augment_texture=True, p_texture=0.,
                              augment_geom=True, p_geom=0., artistic_start=2, artistic_step=1,
                              img_path=FLAGS.img_path, save_log_path=FLAGS.save_log_path,
                              save_sample_path=FLAGS.save_sample_path, save_model_path=FLAGS.save_model_path,
                              test_data=FLAGS.test_data, test_model_path=FLAGS.test_model_path,
                              load_pretrain=False, pre_train_path=FLAGS.pre_train_path)

    if FLAGS.mode == 'TRAIN':
        model.train()
    else:
        model.eval()

if __name__ == '__main__':
    tf.app.run()
