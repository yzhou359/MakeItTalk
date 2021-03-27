import numpy as np
import os
from skimage.color import gray2rgb


def train_val_shuffle_inds_per_epoch(valid_inds, train_inds, train_iter, batch_size, log_path, save_log=True):
    """shuffle image indices for each training epoch and save to log"""
    
    np.random.seed(0)
    num_train_images = len(train_inds)
    num_epochs = int(np.ceil((1. * train_iter) / (1. * num_train_images / batch_size)))+1
    epoch_inds_shuffle = np.zeros((num_epochs, num_train_images)).astype(int)
    img_inds = np.arange(num_train_images)
    for i in range(num_epochs):
        np.random.shuffle(img_inds)
        epoch_inds_shuffle[i, :] = img_inds

    if save_log:
        with open(os.path.join(log_path, "train_val_shuffle_inds.csv"), "wb") as f:
            if valid_inds is not None:
                f.write(b'valid inds\n')
                np.savetxt(f, valid_inds.reshape(1, -1), fmt='%i', delimiter=",")
            f.write(b'train inds\n')
            np.savetxt(f, train_inds.reshape(1, -1), fmt='%i', delimiter=",")
            f.write(b'shuffle inds\n')
            np.savetxt(f, epoch_inds_shuffle, fmt='%i', delimiter=",")

    return epoch_inds_shuffle


def gaussian(x, y, x0, y0, sigma=6):
    return 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def create_gaussian_filter(sigma=6, win_mult=3.5):
    win_size = int(win_mult * sigma)
    x, y = np.mgrid[0:2*win_size+1, 0:2*win_size+1]
    gauss_filt = (8./3)*sigma*gaussian(x, y, win_size, win_size, sigma=sigma)  # same as in ECT
    return gauss_filt


def load_images(img_list, batch_inds, image_size=256, c_dim=3, scale=255):

        """ load images as a numpy array from menpo image list """

        num_inputs = len(batch_inds)
        batch_menpo_images = img_list[batch_inds]

        images = np.zeros([num_inputs, image_size, image_size, c_dim]).astype('float32')

        for ind, img in enumerate(batch_menpo_images):
            if img.n_channels < 3 and c_dim == 3:
                images[ind, :, :, :] = gray2rgb(img.pixels_with_channels_at_back())
            else:
                images[ind, :, :, :] = img.pixels_with_channels_at_back()

        if scale is 255:
            images *= 255
        elif scale is 0:
            images = 2 * images - 1

        return images


# loading functions with pre-allocation and approx heat-map generation


def create_approx_heat_maps_alloc_once(landmarks, maps, gauss_filt=None, win_mult=3.5, num_landmarks=68, image_size=256,
                                       sigma=6):
    """ create heatmaps from input landmarks"""
    maps.fill(0.)

    win_size = int(win_mult * sigma)
    filt_size = 2 * win_size + 1
    landmarks = landmarks.astype(int)

    if gauss_filt is None:
        x_small, y_small = np.mgrid[0:2 * win_size + 1, 0:2 * win_size + 1]
        gauss_filt = (8. / 3) * sigma * gaussian(x_small, y_small, win_size, win_size, sigma=sigma)  # same as in ECT

    for i in range(num_landmarks):

        min_row = landmarks[i, 0] - win_size
        max_row = landmarks[i, 0] + win_size + 1
        min_col = landmarks[i, 1] - win_size
        max_col = landmarks[i, 1] + win_size + 1

        if min_row < 0:
            min_row_gap = -1 * min_row
            min_row = 0
        else:
            min_row_gap = 0

        if min_col < 0:
            min_col_gap = -1 * min_col
            min_col = 0
        else:
            min_col_gap = 0

        if max_row > image_size:
            max_row_gap = max_row - image_size
            max_row = image_size
        else:
            max_row_gap = 0

        if max_col > image_size:
            max_col_gap = max_col - image_size
            max_col = image_size
        else:
            max_col_gap = 0

        maps[min_row:max_row, min_col:max_col, i] =\
            gauss_filt[min_row_gap:filt_size - 1 * max_row_gap, min_col_gap:filt_size - 1 * max_col_gap]


def load_images_landmarks_approx_maps_alloc_once(
        img_list, batch_inds, images, maps_small, maps, landmarks, image_size=256, num_landmarks=68,
        scale=255, gauss_filt_large=None, gauss_filt_small=None, win_mult=3.5, sigma=6, save_landmarks=False):

    """ load images and gt landmarks from menpo image list, and create matching heatmaps """

    batch_menpo_images = img_list[batch_inds]
    c_dim = images.shape[-1]
    grp_name = batch_menpo_images[0].landmarks.group_labels[0]

    win_size_large = int(win_mult * sigma)
    win_size_small = int(win_mult * (1.*sigma/4))

    if gauss_filt_small is None:
        x_small, y_small = np.mgrid[0:2 * win_size_small + 1, 0:2 * win_size_small + 1]
        gauss_filt_small = (8. / 3) * (1.*sigma/4) * gaussian(
            x_small, y_small, win_size_small, win_size_small, sigma=1.*sigma/4)  # same as in ECT
    if gauss_filt_large is None:
        x_large, y_large = np.mgrid[0:2 * win_size_large + 1, 0:2 * win_size_large + 1]
        gauss_filt_large = (8. / 3) * sigma * gaussian(x_large, y_large, win_size_large, win_size_large, sigma=sigma)  # same as in ECT

    for ind, img in enumerate(batch_menpo_images):
        if img.n_channels < 3 and c_dim == 3:
            images[ind, :, :, :] = gray2rgb(img.pixels_with_channels_at_back())
        else:
            images[ind, :, :, :] = img.pixels_with_channels_at_back()

        lms = img.landmarks[grp_name].points
        lms = np.minimum(lms, image_size - 1)
        create_approx_heat_maps_alloc_once(
            landmarks=lms, maps=maps[ind, :, :, :], gauss_filt=gauss_filt_large, win_mult=win_mult,
            num_landmarks=num_landmarks, image_size=image_size, sigma=sigma)

        lms_small = img.resize([image_size / 4, image_size / 4]).landmarks[grp_name].points
        lms_small = np.minimum(lms_small, image_size / 4 - 1)
        create_approx_heat_maps_alloc_once(
            landmarks=lms_small, maps=maps_small[ind, :, :, :], gauss_filt=gauss_filt_small, win_mult=win_mult,
            num_landmarks=num_landmarks, image_size=image_size / 4, sigma=1. * sigma / 4)

        if save_landmarks:
            landmarks[ind, :, :] = lms

    if scale is 255:
        images *= 255
    elif scale is 0:
        images = 2 * images - 1
