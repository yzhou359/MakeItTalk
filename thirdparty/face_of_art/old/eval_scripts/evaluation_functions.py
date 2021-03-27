import tensorflow as tf
from menpofit.visualize import plot_cumulative_error_distribution
from menpofit.error import compute_cumulative_error
from scipy.integrate import simps
from menpo_functions import load_menpo_image_list, load_bb_dictionary
from logging_functions import *
from data_loading_functions import *
from time import time
import sys
from PyQt5 import QtWidgets
qapp=QtWidgets.QApplication([''])


def load_menpo_test_list(img_dir, test_data='full', image_size=256, margin=0.25, bb_type='gt'):
    mode = 'TEST'
    bb_dir = os.path.join(img_dir, 'Bounding_Boxes')
    bb_dictionary = load_bb_dictionary(bb_dir, mode, test_data=test_data)
    img_menpo_list = load_menpo_image_list(
        img_dir=img_dir, train_crop_dir=None, img_dir_ns=None, mode=mode, bb_dictionary=bb_dictionary,
        image_size=image_size, margin=margin,
        bb_type=bb_type, test_data=test_data, augment_basic=False, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0)
    return img_menpo_list


def evaluate_heatmap_fusion_network(model_path, img_path, test_data, batch_size=10, image_size=256, margin=0.25,
                                    bb_type='gt', c_dim=3, scale=1, num_landmarks=68, debug=False,
                                    debug_data_size=20):
    t = time()
    from deep_heatmaps_model_fusion_net import DeepHeatmapsModel
    import logging
    logging.getLogger('tensorflow').disabled = True

    # load test image menpo list

    test_menpo_img_list = load_menpo_test_list(
        img_path, test_data=test_data, image_size=image_size, margin=margin, bb_type=bb_type)

    if debug:
        test_menpo_img_list = test_menpo_img_list[:debug_data_size]
        print ('\n*** FUSION NETWORK: calculating normalized mean error on: ' + test_data +
               ' set (%d images - debug mode) ***' % debug_data_size)
    else:
        print ('\n*** FUSION NETWORK: calculating normalized mean error on: ' + test_data + ' set (%d images) ***' %
               (len(test_menpo_img_list)))

    # create heatmap model

    tf.reset_default_graph()

    model = DeepHeatmapsModel(mode='TEST', batch_size=batch_size, image_size=image_size, c_dim=c_dim,
                              num_landmarks=num_landmarks, img_path=img_path, test_model_path=model_path,
                              test_data=test_data, menpo_verbose=False)

    # add placeholders
    model.add_placeholders()
    # build model
    model.build_model()
    # create loss ops
    model.create_loss_ops()

    num_batches = int(1. * len(test_menpo_img_list) / batch_size)
    if num_batches == 0:
        batch_size = len(test_menpo_img_list)
        num_batches = 1

    reminder = len(test_menpo_img_list) - num_batches * batch_size
    num_batches_reminder = num_batches + 1 * (reminder > 0)
    img_inds = np.arange(len(test_menpo_img_list))

    with tf.Session() as session:

        # load trained parameters
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        print ('\nnum batches: ' + str(num_batches_reminder))

        err = []
        for j in range(num_batches):
            print ('batch %d / %d ...' % (j + 1, num_batches_reminder))
            batch_inds = img_inds[j * batch_size:(j + 1) * batch_size]

            batch_images, _, batch_landmarks_gt = load_images_landmarks(
                test_menpo_img_list, batch_inds=batch_inds, image_size=image_size,
                c_dim=c_dim, num_landmarks=num_landmarks, scale=scale)

            batch_maps_pred = session.run(model.pred_hm_f, {model.images: batch_images})

            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_pred, batch_size=batch_size, image_size=image_size, num_landmarks=num_landmarks)

            batch_err = session.run(
                model.nme_per_image, {model.lms: batch_landmarks_gt, model.pred_lms: batch_pred_landmarks})
            err = np.hstack((err, batch_err))

        if reminder > 0:
            print ('batch %d / %d ...' % (j + 2, num_batches_reminder))
            reminder_inds = img_inds[-reminder:]

            batch_images, _, batch_landmarks_gt = load_images_landmarks(
                test_menpo_img_list, batch_inds=reminder_inds, image_size=image_size,
                c_dim=c_dim, num_landmarks=num_landmarks, scale=scale)

            batch_maps_pred = session.run(model.pred_hm_f, {model.images: batch_images})

            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_pred, batch_size=reminder, image_size=image_size, num_landmarks=num_landmarks)

            batch_err = session.run(
                model.nme_per_image, {model.lms: batch_landmarks_gt, model.pred_lms: batch_pred_landmarks})
            err = np.hstack((err, batch_err))

        print ('\ndone!')
        print ('run time: ' + str(time() - t))

    return err


def evaluate_heatmap_primary_network(model_path, img_path, test_data, batch_size=10, image_size=256, margin=0.25,
                                     bb_type='gt', c_dim=3, scale=1, num_landmarks=68, debug=False,
                                     debug_data_size=20):
    t = time()
    from deep_heatmaps_model_primary_net import DeepHeatmapsModel
    import logging
    logging.getLogger('tensorflow').disabled = True

    # load test image menpo list

    test_menpo_img_list = load_menpo_test_list(
        img_path, test_data=test_data, image_size=image_size, margin=margin, bb_type=bb_type)

    if debug:
        test_menpo_img_list = test_menpo_img_list[:debug_data_size]
        print ('\n*** PRIMARY NETWORK: calculating normalized mean error on: ' + test_data +
               ' set (%d images - debug mode) ***' % debug_data_size)
    else:
        print ('\n*** PRIMARY NETWORK: calculating normalized mean error on: ' + test_data +
               ' set (%d images) ***' % (len(test_menpo_img_list)))

    # create heatmap model

    tf.reset_default_graph()

    model = DeepHeatmapsModel(mode='TEST', batch_size=batch_size, image_size=image_size, c_dim=c_dim,
                              num_landmarks=num_landmarks, img_path=img_path, test_model_path=model_path,
                              test_data=test_data, menpo_verbose=False)

    # add placeholders
    model.add_placeholders()
    # build model
    model.build_model()
    # create loss ops
    model.create_loss_ops()

    num_batches = int(1. * len(test_menpo_img_list) / batch_size)
    if num_batches == 0:
        batch_size = len(test_menpo_img_list)
        num_batches = 1

    reminder = len(test_menpo_img_list) - num_batches * batch_size
    num_batches_reminder = num_batches + 1 * (reminder > 0)
    img_inds = np.arange(len(test_menpo_img_list))

    with tf.Session() as session:

        # load trained parameters
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        print ('\nnum batches: ' + str(num_batches_reminder))

        err = []
        for j in range(num_batches):
            print ('batch %d / %d ...' % (j + 1, num_batches_reminder))
            batch_inds = img_inds[j * batch_size:(j + 1) * batch_size]

            batch_images, _, batch_landmarks_gt = load_images_landmarks(
                test_menpo_img_list, batch_inds=batch_inds, image_size=image_size,
                c_dim=c_dim, num_landmarks=num_landmarks, scale=scale)

            batch_maps_small_pred = session.run(model.pred_hm_p, {model.images: batch_images})

            batch_maps_small_pred = zoom(batch_maps_small_pred, zoom=[1, 4, 4, 1], order=1)  # NN interpolation

            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, batch_size=batch_size, image_size=image_size,
                num_landmarks=num_landmarks)

            batch_err = session.run(
                model.nme_per_image, {model.lms_small: batch_landmarks_gt, model.pred_lms_small: batch_pred_landmarks})
            err = np.hstack((err, batch_err))

        if reminder > 0:
            print ('batch %d / %d ...' % (j + 2, num_batches_reminder))
            reminder_inds = img_inds[-reminder:]

            batch_images, _, batch_landmarks_gt = load_images_landmarks(
                test_menpo_img_list, batch_inds=reminder_inds, image_size=image_size,
                c_dim=c_dim, num_landmarks=num_landmarks, scale=scale)

            batch_maps_small_pred = session.run(model.pred_hm_p, {model.images: batch_images})

            batch_maps_small_pred = zoom(batch_maps_small_pred, zoom=[1, 4, 4, 1], order=1)  # NN interpolation

            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, batch_size=reminder, image_size=image_size,
                num_landmarks=num_landmarks)

            batch_err = session.run(
                model.nme_per_image, {model.lms_small: batch_landmarks_gt, model.pred_lms_small: batch_pred_landmarks})
            err = np.hstack((err, batch_err))

        print ('\ndone!')
        print ('run time: ' + str(time() - t))

    return err


def evaluate_heatmap_network(model_path, network_type, img_path, test_data, batch_size=10, image_size=256, margin=0.25,
                                     bb_type='gt', c_dim=3, scale=1, num_landmarks=68, debug=False,
                                     debug_data_size=20):

    if network_type.lower() == 'fusion':
        return evaluate_heatmap_fusion_network(
            model_path=model_path, img_path=img_path, test_data=test_data, batch_size=batch_size, image_size=image_size,
            margin=margin, bb_type=bb_type, c_dim=c_dim, scale=scale, num_landmarks=num_landmarks, debug=debug,
            debug_data_size=debug_data_size)
    elif network_type.lower() == 'primary':
        return evaluate_heatmap_primary_network(
            model_path=model_path, img_path=img_path, test_data=test_data, batch_size=batch_size, image_size=image_size,
            margin=margin, bb_type=bb_type, c_dim=c_dim, scale=scale, num_landmarks=num_landmarks, debug=debug,
            debug_data_size=debug_data_size)
    else:
        sys.exit('\n*** Error: please choose a valid network type: Fusion/Primary ***')


def AUC(errors, max_error, step_error=0.0001):
    x_axis = list(np.arange(0., max_error + step_error, step_error))
    ced = np.array(compute_cumulative_error(errors, x_axis))
    return simps(ced, x=x_axis) / max_error, 1. - ced[-1]


def print_nme_statistics(
        errors, model_path, network_type, test_data, max_error=0.08, log_path='', save_log=True, plot_ced=True,
        norm='interocular distance'):
    auc, failures = AUC(errors, max_error=max_error)

    print ("\n****** NME statistics for " + network_type + " Network ******\n")
    print ("* model path: " + model_path)
    print ("* dataset: " + test_data + ' set')

    print ("\n* Normalized mean error (percentage of "+norm+"): %.2f" % (100 * np.mean(errors)))
    print ("\n* AUC @ %.2f: %.2f" % (max_error, 100 * auc))
    print ("\n* failure rate @ %.2f: %.2f" % (max_error, 100 * failures) + '%')

    if plot_ced:
        plt.figure()
        plt.yticks(np.linspace(0, 1, 11))
        plot_cumulative_error_distribution(
            list(errors),
            legend_entries=[network_type],
            marker_style=['s'],
            marker_size=7,
            x_label='Normalised Point-to-Point Error\n('+norm+')\n*' + test_data + ' set*',
        )

    if save_log:
        with open(os.path.join(log_path, network_type.lower() + "_nme_statistics_on_" + test_data + "_set.txt"),
                  "wb") as f:
            f.write(b"************************************************")
            f.write(("\n****** NME statistics for " + str(network_type) + " Network ******\n").encode())
            f.write(b"************************************************")
            f.write(("\n\n* model path: " + str(model_path)).encode())
            f.write(("\n\n* dataset: " + str(test_data) + ' set').encode())
            f.write(b"\n\n* Normalized mean error (percentage of "+norm+"): %.2f" % (100 * np.mean(errors)))
            f.write(b"\n\n* AUC @ %.2f: %.2f" % (max_error, 100 * auc))
            f.write(("\n\n* failure rate @ %.2f: %.2f" % (max_error, 100 * failures) + '%').encode())
        if plot_ced:
            plt.savefig(os.path.join(log_path, network_type.lower() + '_nme_ced_on_' + test_data + '_set.png'),
                        bbox_inches='tight')
            plt.close()

        print ('\nlog path: ' + log_path)


def print_ced_compare_methods(
        method_errors,method_names,test_data,log_path='', save_log=True, norm='interocular distance'):
    plt.yticks(np.linspace(0, 1, 11))
    plot_cumulative_error_distribution(
        [list(err) for err in list(method_errors)],
        legend_entries=list(method_names),
        marker_style=['s'],
        marker_size=7,
        x_label='Normalised Point-to-Point Error\n('+norm+')\n*'+test_data+' set*'
    )
    if save_log:
        plt.savefig(os.path.join(log_path,'nme_ced_on_'+test_data+'_set.png'), bbox_inches='tight')
        print ('ced plot path: ' + os.path.join(log_path,'nme_ced_on_'+test_data+'_set.png'))
        plt.close()