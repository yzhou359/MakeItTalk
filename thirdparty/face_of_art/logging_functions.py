import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def print_training_params_to_file(init_locals):
    """save param log file"""

    del init_locals['self']
    with open(os.path.join(init_locals['save_log_path'], 'Training_Parameters.txt'), 'w') as f:
        f.write('Training Parameters:\n\n')
        for key, value in init_locals.items():
            f.write('* %s: %s\n' % (key, value))


def heat_maps_to_landmarks(maps, image_size=256, num_landmarks=68):
    """find landmarks from heatmaps (arg max on each map)"""

    landmarks = np.zeros((num_landmarks,2)).astype('float32')

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))

    return landmarks


def heat_maps_to_landmarks_alloc_once(maps, landmarks, image_size=256, num_landmarks=68):
    """find landmarks from heatmaps (arg max on each map) with pre-allocation"""

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))


def batch_heat_maps_to_landmarks_alloc_once(batch_maps, batch_landmarks, batch_size, image_size=256, num_landmarks=68):
    """find landmarks from heatmaps (arg max on each map) - for multiple images"""

    for i in range(batch_size):
        heat_maps_to_landmarks_alloc_once(
            maps=batch_maps[i, :, :, :], landmarks=batch_landmarks[i, :, :], image_size=image_size,
            num_landmarks=num_landmarks)


def normalize_map(map_in):
    map_min = map_in.min()
    return (map_in - map_min) / (map_in.max() - map_min)


def map_to_rgb(map_gray):
    cmap = plt.get_cmap('jet')
    rgba_map_image = cmap(map_gray)
    map_rgb = np.delete(rgba_map_image, 3, 2) * 255
    return map_rgb


def create_img_with_landmarks(image, landmarks, image_size=256, num_landmarks=68, scale=255, circle_size=2):
    """add landmarks to a face image"""
    image = image.reshape(image_size, image_size, -1)

    if scale is 0:
        image = 127.5 * (image + 1)
    elif scale is 1:
        image *= 255

    landmarks = landmarks.reshape(num_landmarks, 2)
    landmarks = np.clip(landmarks, 0, image_size-1)

    for (y, x) in landmarks.astype('int'):
        cv2.circle(image, (x, y), circle_size, (255, 0, 0), -1)

    return image


def heat_maps_to_image(maps, landmarks=None, image_size=256, num_landmarks=68):
    """create one image from multiple heatmaps"""

    if landmarks is None:
        landmarks = heat_maps_to_landmarks(maps, image_size=image_size, num_landmarks=num_landmarks)

    x, y = np.mgrid[0:image_size, 0:image_size]

    pixel_dist = np.sqrt(
        np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

    nn_landmark = np.argmin(pixel_dist, 2)

    map_image = maps[x, y, nn_landmark]
    map_image = (map_image-map_image.min())/(map_image.max()-map_image.min())  # normalize for visualization

    return map_image


def merge_images_landmarks_maps_gt(images, maps, maps_gt, landmarks=None, image_size=256, num_landmarks=68,
                                   num_samples=9, scale=255, circle_size=2, fast=False):
    """create image for log - containing input face images, predicted heatmaps and GT heatmaps (if exists)"""

    images = images[:num_samples]
    if maps.shape[1] is not image_size:
        images = zoom(images, (1, 0.25, 0.25, 1))
        image_size /= 4
        image_size=int(image_size)
    if maps_gt is not None:
        if maps_gt.shape[1] is not image_size:
            maps_gt = zoom(maps_gt, (1, 0.25, 0.25, 1))

    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    if maps_gt is None:
        merged = np.zeros([row * image_size, row * image_size * 2, 3])
    else:
        merged = np.zeros([row * image_size, row * image_size * 3, 3])

    for idx, img in enumerate(images):
        i = idx // row
        j = idx % row

        if landmarks is None:
            img_landmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size,
                                                   num_landmarks=num_landmarks)
        else:
            img_landmarks = landmarks[idx]

        if fast:
            map_image = np.amax(maps[idx, :, :, :], 2)
            map_image = (map_image - map_image.min()) / (map_image.max() - map_image.min())
        else:
            map_image = heat_maps_to_image(maps[idx, :, :, :], img_landmarks, image_size=image_size,
                                           num_landmarks=num_landmarks)
        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        img = create_img_with_landmarks(img, img_landmarks, image_size, num_landmarks, scale=scale,
                                        circle_size=circle_size)

        if maps_gt is not None:
            if fast:
                map_gt_image = np.amax(maps_gt[idx, :, :, :], 2)
                map_gt_image = (map_gt_image - map_gt_image.min()) / (map_gt_image.max() - map_gt_image.min())
            else:
                map_gt_image = heat_maps_to_image(maps_gt[idx, :, :, :], image_size=image_size,
                                                  num_landmarks=num_landmarks)
            rgba_map_gt_image = cmap(map_gt_image)
            map_gt_image = np.delete(rgba_map_gt_image, 3, 2) * 255

            merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] = img
            merged[i * image_size:(i + 1) * image_size, (j * 3 + 1) * image_size:(j * 3 + 2) * image_size,
            :] = map_image
            merged[i * image_size:(i + 1) * image_size, (j * 3 + 2) * image_size:(j * 3 + 3) * image_size,
            :] = map_gt_image
        else:
            merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = img
            merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size,:] = map_image

    return merged


def map_comapre_channels(images, maps1, maps2, image_size=64, num_landmarks=68, scale=255):
    """create image for log - present one face image, along with all its heatmaps (one for each landmark)"""

    map1 = maps1[0]
    if maps2 is not None:
        map2 = maps2[0]
    image = images[0]

    if image.shape[0] is not image_size:
        image = zoom(image, (0.25, 0.25, 1))
    if scale is 1:
        image *= 255
    elif scale is 0:
        image = 127.5 * (image + 1)

    row = np.ceil(np.sqrt(num_landmarks)).astype(np.int64)
    if maps2 is not None:
        merged = np.zeros([row * image_size, row * image_size * 2, 3])
    else:
        merged = np.zeros([row * image_size, row * image_size, 3])

    for idx in range(num_landmarks):
        i = idx // row
        j = idx % row
        channel_map = map_to_rgb(normalize_map(map1[:, :, idx]))
        if maps2 is not None:
            channel_map2 = map_to_rgb(normalize_map(map2[:, :, idx]))
            merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] =\
                channel_map
            merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] =\
                channel_map2
        else:
            merged[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, :] = channel_map

    i = (idx + 1) // row
    j = (idx + 1) % row
    if maps2 is not None:
        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = image
    else:
        merged[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size, :] = image
    return merged

