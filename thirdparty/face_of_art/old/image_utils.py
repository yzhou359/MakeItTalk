import numpy as np
import os
from scipy.io import loadmat
import cv2
from menpo.shape.pointcloud import PointCloud
from menpo.transform import ThinPlateSplines
import menpo.io as mio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from glob import glob
from deformation_functions import *

'''********* bounding box and image loading functions *********'''


def center_margin_bb(bb, img_bounds, margin=0.25):
    bb_size = ([bb[0, 2] - bb[0, 0], bb[0, 3] - bb[0, 1]])
    margins = (np.max(bb_size) * (1 + margin) - bb_size) / 2

    bb_new = np.zeros_like(bb)
    bb_new[0, 0] = np.maximum(bb[0, 0] - margins[0], 0)
    bb_new[0, 2] = np.minimum(bb[0, 2] + margins[0], img_bounds[1])
    bb_new[0, 1] = np.maximum(bb[0, 1] - margins[1], 0)
    bb_new[0, 3] = np.minimum(bb[0, 3] + margins[1], img_bounds[0])
    return bb_new


def load_bb_files(bb_file_dirs):
    bb_files_dict = {}
    for bb_file in bb_file_dirs:
        bb_mat = loadmat(bb_file)['bounding_boxes']
        num_imgs = np.max(bb_mat.shape)
        for i in range(num_imgs):
            name = bb_mat[0][i][0][0][0][0]
            bb_init = bb_mat[0][i][0][0][1] - 1  # matlab indicies
            bb_gt = bb_mat[0][i][0][0][2] - 1  # matlab indicies
            if str(name) in bb_files_dict.keys():
                print str(name), 'already loaded from: ', bb_file
            bb_files_dict[str(name)] = (bb_init, bb_gt)
    return bb_files_dict


def load_bb_dictionary(bb_dir, mode, test_data='full'):
    if mode == 'TRAIN':
        bb_dirs = \
            ['bounding_boxes_afw.mat', 'bounding_boxes_helen_trainset.mat', 'bounding_boxes_lfpw_trainset.mat']
    else:
        if test_data == 'common':
            bb_dirs = \
                ['bounding_boxes_helen_testset.mat', 'bounding_boxes_lfpw_testset.mat']
        elif test_data == 'challenging':
            bb_dirs = ['bounding_boxes_ibug.mat']
        elif test_data == 'full':
            bb_dirs = \
                ['bounding_boxes_ibug.mat', 'bounding_boxes_helen_testset.mat', 'bounding_boxes_lfpw_testset.mat']
        elif test_data == 'training':
            bb_dirs = \
                ['bounding_boxes_afw.mat', 'bounding_boxes_helen_trainset.mat', 'bounding_boxes_lfpw_trainset.mat']
        else:
            bb_dirs=None

    if mode == 'TEST' and test_data not in ['full', 'challenging', 'common', 'training']:
        bb_files_dict = None
    else:
        bb_dirs = [os.path.join(bb_dir, dataset) for dataset in bb_dirs]
        bb_files_dict = load_bb_files(bb_dirs)

    return bb_files_dict


def crop_to_face_image(img, bb_dictionary=None, gt=True, margin=0.25, image_size=256):
    name = img.path.name
    img_bounds = img.bounds()[1]

    if bb_dictionary is None:
        bb_menpo = img.landmarks['PTS'].bounding_box().points
        bb = np.array([[bb_menpo[0, 1], bb_menpo[0, 0], bb_menpo[2, 1], bb_menpo[2, 0]]])
    else:
        if gt:
            bb = bb_dictionary[name][1]  # ground truth
        else:
            bb = bb_dictionary[name][0]  # init from face detector

    bb = center_margin_bb(bb, img_bounds, margin=margin)

    bb_pointcloud = PointCloud(np.array([[bb[0, 1], bb[0, 0]],
                                         [bb[0, 3], bb[0, 0]],
                                         [bb[0, 3], bb[0, 2]],
                                         [bb[0, 1], bb[0, 2]]]))

    face_crop = img.crop_to_pointcloud(bb_pointcloud).resize([image_size, image_size])

    return face_crop


def augment_face_image(img, image_size=256, crop_size=248, angle_range=30, flip=True):

    # taken from MDM
    jaw_indices = np.arange(0, 17)
    lbrow_indices = np.arange(17, 22)
    rbrow_indices = np.arange(22, 27)
    upper_nose_indices = np.arange(27, 31)
    lower_nose_indices = np.arange(31, 36)
    leye_indices = np.arange(36, 42)
    reye_indices = np.arange(42, 48)
    outer_mouth_indices = np.arange(48, 60)
    inner_mouth_indices = np.arange(60, 68)

    mirrored_parts_68 = np.hstack([
        jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
        upper_nose_indices, lower_nose_indices[::-1],
        np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
        np.roll(outer_mouth_indices[::-1], 7),
        np.roll(inner_mouth_indices[::-1], 5)
    ])

    def mirror_landmarks_68(lms, im_size):
        return PointCloud(abs(np.array([0, im_size[1]]) - lms.as_vector(
        ).reshape(-1, 2))[mirrored_parts_68])

    def mirror_image(im):
        im = im.copy()
        im.pixels = im.pixels[..., ::-1].copy()

        for group in im.landmarks:
            lms = im.landmarks[group]
            if lms.points.shape[0] == 68:
                im.landmarks[group] = mirror_landmarks_68(lms, im.shape)

        return im

    lim = image_size - crop_size
    min_crop_inds = np.random.randint(0, lim, 2)
    max_crop_inds = min_crop_inds + crop_size
    flip_rand = np.random.random() > 0.5
    rot_angle = 2 * angle_range * np.random.random_sample() - angle_range

    if flip and flip_rand:
        rand_crop = img.crop(min_crop_inds, max_crop_inds)
        rand_crop = mirror_image(rand_crop)
        rand_crop = rand_crop.rotate_ccw_about_centre(rot_angle).resize([image_size, image_size])

    else:
        rand_crop = img.crop(min_crop_inds, max_crop_inds). \
            rotate_ccw_about_centre(rot_angle).resize([image_size, image_size])

    return rand_crop


def load_menpo_image_list(img_dir, mode, bb_dictionary=None, image_size=256, margin=0.25, bb_type='gt',
                          test_data='full', augment=True):
    def crop_to_face_image_gt(img, bb_dictionary=bb_dictionary, margin=margin, image_size=image_size):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img, bb_dictionary=bb_dictionary, margin=margin, image_size=image_size):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    if mode is 'TRAIN':
        img_set_dir = os.path.join(img_dir, 'training_set')

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')

    image_menpo_list = mio.import_images(img_set_dir, verbose=True)

    if bb_type is 'gt':
        face_crop_image_list = image_menpo_list.map(crop_to_face_image_gt)
    else:
        face_crop_image_list = image_menpo_list.map(crop_to_face_image_init)

    if mode is 'TRAIN' and augment:
        out_image_list = face_crop_image_list.map(augment_face_image)
    else:
        out_image_list = face_crop_image_list

    return out_image_list


def augment_menpo_img_ns(img, img_dir_ns, p_ns=0):
    img = img.copy()
    texture_aug = p_ns > 0.5
    if texture_aug:
        ns_augs = glob(os.path.join(img_dir_ns, img.path.name.split('.')[0] + '*'))
        num_augs = len(ns_augs)
        if num_augs > 1:
            ns_ind = np.random.randint(1, num_augs)
            ns_aug = mio.import_image(ns_augs[ns_ind])
            ns_pixels = ns_aug.pixels
            img.pixels = ns_pixels
    return img


def augment_menpo_img_geom(img, p_geom=0):
    img = img.copy()
    if p_geom > 0.5:
        lms_geom_warp=deform_face_geometric_style(img.landmarks['PTS'].points.copy(),p_scale=p_geom,p_shift=p_geom)
        img = warp_face_image_tps(img,PointCloud(lms_geom_warp))
    return img


def warp_face_image_tps(img,new_shape):
    tps = ThinPlateSplines(new_shape, img.landmarks['PTS'])
    img_warp=img.warp_to_shape(img.shape,tps)
    img_warp.landmarks['PTS']=new_shape
    return img_warp


def load_menpo_image_list_artistic_aug(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full',augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns(img, img_dir_ns, p_ns=1. * (np.random.rand() <= p_texture))

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom(img, p_geom=1. * (np.random.rand() <= p_geom))

    if mode is 'TRAIN':
        img_set_dir = os.path.join(img_dir, train_crop_dir)
        out_image_list = mio.import_images(img_set_dir, verbose=True)

        if augment_texture:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        out_image_list = mio.import_images(img_set_dir, verbose=True)
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)

    return out_image_list


def reload_img_menpo_list_artistic_aug_train(
        img_dir, train_crop_dir, img_dir_ns, mode, train_inds, image_size=256,
        augment_basic=True, augment_texture=False, p_texture=0, augment_geom=False, p_geom=0):

    img_menpo_list = load_menpo_image_list_artistic_aug(
        img_dir=img_dir, train_crop_dir=train_crop_dir, img_dir_ns=img_dir_ns, mode=mode,image_size=image_size,
        augment_basic=augment_basic, augment_texture=augment_texture, p_texture=p_texture, augment_geom=augment_geom,
        p_geom=p_geom)

    img_menpo_list_train = img_menpo_list[train_inds]

    return img_menpo_list_train


'''********* heat-maps and image loading functions *********'''


# look for: ECT-FaceAlignment/caffe/src/caffe/layers/data_heatmap.cpp
def gaussian(x, y, x0, y0, sigma=6):
    return 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def create_heat_maps(landmarks, num_landmarks=68, image_size=256, sigma=6):

    x, y = np.mgrid[0:image_size, 0:image_size]

    maps = np.zeros((image_size, image_size, num_landmarks))

    for i in range(num_landmarks):
        out = gaussian(x, y, landmarks[i,0], landmarks[i,1], sigma=sigma)
        maps[:, :, i] = (8./3)*sigma*out  # copied from ECT

    return maps


def load_data(img_list, batch_inds, image_size=256, c_dim=3, num_landmarks=68 , sigma=6, scale='255',
              save_landmarks=False, primary=False):

    num_inputs = len(batch_inds)
    batch_menpo_images = img_list[batch_inds]

    images = np.zeros([num_inputs, image_size, image_size, c_dim]).astype('float32')
    maps_small = np.zeros([num_inputs, image_size/4, image_size/4, num_landmarks]).astype('float32')

    if primary:
        maps = None
    else:
        maps = np.zeros([num_inputs, image_size, image_size, num_landmarks]).astype('float32')

    if save_landmarks:
        landmarks = np.zeros([num_inputs, num_landmarks, 2]).astype('float32')
    else:
        landmarks = None

    for ind, img in enumerate(batch_menpo_images):

        images[ind, :, :, :] = np.rollaxis(img.pixels, 0, 3)

        if primary:
            lms = img.resize([image_size/4,image_size/4]).landmarks['PTS'].points
            maps_small[ind, :, :, :] = create_heat_maps(lms, num_landmarks, image_size/4, sigma)
        else:
            lms = img.landmarks['PTS'].points
            maps[ind, :, :, :] = create_heat_maps(lms, num_landmarks, image_size, sigma)
            maps_small[ind, :, :, :]=zoom(maps[ind, :, :, :],(0.25,0.25,1))

        if save_landmarks:
            landmarks[ind, :, :] = lms

    if scale is '255':
        images *= 255  # SAME AS ECT?
    elif scale is '0':
        images = 2 * images - 1

    return images, maps, maps_small, landmarks


def heat_maps_to_image(maps, landmarks=None, image_size=256, num_landmarks=68):

    if landmarks is None:
        landmarks = heat_maps_to_landmarks(maps, image_size=image_size, num_landmarks=num_landmarks)

    x, y = np.mgrid[0:image_size, 0:image_size]

    pixel_dist = np.sqrt(
        np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

    nn_landmark = np.argmin(pixel_dist, 2)

    map_image = maps[x, y, nn_landmark]
    map_image = (map_image-map_image.min())/(map_image.max()-map_image.min())  # normalize for visualization

    return map_image


def heat_maps_to_landmarks(maps, image_size=256, num_landmarks=68):

    landmarks = np.zeros((num_landmarks,2)).astype('float32')

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))

    return landmarks


def batch_heat_maps_to_landmarks(batch_maps, batch_size, image_size=256, num_landmarks=68):
    batch_landmarks = np.zeros((batch_size,num_landmarks, 2)).astype('float32')
    for i in range(batch_size):
        batch_landmarks[i,:,:]=heat_maps_to_landmarks(
            batch_maps[i,:,:,:], image_size=image_size, num_landmarks=num_landmarks)

    return batch_landmarks


def print_training_params_to_file(init_locals):
    del init_locals['self']
    with open(os.path.join(init_locals['save_log_path'], 'Training_Parameters.txt'), 'w') as f:
        f.write('Training Parameters:\n\n')
        for key, value in init_locals.items():
            f.write('* %s: %s\n' % (key, value))


def create_img_with_landmarks(image, landmarks, image_size=256, num_landmarks=68, scale='255', circle_size=2):
    image = image.reshape(image_size, image_size, -1)

    if scale is '0':
        image = 127.5 * (image + 1)
    elif scale is '1':
        image *= 255

    landmarks = landmarks.reshape(num_landmarks, 2)
    landmarks = np.clip(landmarks, 0, image_size)

    for (y, x) in landmarks.astype('int'):
        cv2.circle(image, (x, y), circle_size, (255, 0, 0), -1)

    return image


def merge_images_landmarks_maps(images, maps, image_size=256, num_landmarks=68, num_samples=9, scale='255',
                                circle_size=2):
    images = images[:num_samples]
    if maps.shape[1] is not image_size:
        images = zoom(images, (1, 0.25, 0.25, 1))
        image_size /= 4
    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 2, 3])

    for idx, img in enumerate(images):
        i = idx // row
        j = idx % row

        img_lamdmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)
        map_image = heat_maps_to_image(maps[idx, :, :, :], img_lamdmarks, image_size=image_size,
                                       num_landmarks=num_landmarks)

        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        img = create_img_with_landmarks(img, img_lamdmarks, image_size, num_landmarks, scale=scale,
                                        circle_size=circle_size)

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = img
        merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = map_image

    return merged


def merge_compare_maps(maps_small, maps, image_size=64, num_landmarks=68, num_samples=9):

    maps_small = maps_small[:num_samples]
    maps = maps[:num_samples]

    if maps_small.shape[1] is not image_size:
        image_size = maps_small.shape[1]

    if maps.shape[1] is not maps_small.shape[1]:
        maps_rescale = zoom(maps, (1, 0.25, 0.25, 1))
    else:
        maps_rescale = maps

    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 2, 3])

    for idx, map_small in enumerate(maps_small):
        i = idx // row
        j = idx % row

        map_image_small = heat_maps_to_image(map_small, image_size=image_size, num_landmarks=num_landmarks)
        map_image = heat_maps_to_image(maps_rescale[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)

        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        rgba_map_image_small = cmap(map_image_small)
        map_image_small = np.delete(rgba_map_image_small, 3, 2) * 255

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = map_image_small
        merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = map_image

    return merged


def normalize_map(map_in):
    return (map_in - map_in.min()) / (map_in.max() - map_in.min())


def map_to_rgb(map_gray):
    cmap = plt.get_cmap('jet')
    rgba_map_image = cmap(map_gray)
    map_rgb = np.delete(rgba_map_image, 3, 2) * 255
    return map_rgb


def load_art_data(img_list, batch_inds, image_size=256, c_dim=3, scale='255'):

    num_inputs = len(batch_inds)
    batch_menpo_images = img_list[batch_inds]

    images = np.zeros([num_inputs, image_size, image_size, c_dim]).astype('float32')

    for ind, img in enumerate(batch_menpo_images):
        images[ind, :, :, :] = np.rollaxis(img.pixels, 0, 3)

    if scale is '255':
        images *= 255  # SAME AS ECT?
    elif scale is '0':
        images = 2 * images - 1

    return images


def merge_images_landmarks_maps_gt(images, maps, maps_gt, image_size=256, num_landmarks=68, num_samples=9, scale='255',
                                   circle_size=2, test_data='full', fast=False):
    images = images[:num_samples]
    if maps.shape[1] is not image_size:
        images = zoom(images, (1, 0.25, 0.25, 1))
        image_size /= 4
    if maps_gt.shape[1] is not image_size:
        maps_gt = zoom(maps_gt, (1, 0.25, 0.25, 1))

    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 3, 3])

    if fast:
        maps_gt_images = np.amax(maps_gt, 3)
        maps_images = np.amax(maps, 3)

    for idx, img in enumerate(images):
        i = idx // row
        j = idx % row

        img_landmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)

        if fast:
            map_image = maps_images[idx]
        else:
            map_image = heat_maps_to_image(maps[idx, :, :, :], img_landmarks, image_size=image_size,
                                           num_landmarks=num_landmarks)
        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        if test_data not in ['full', 'challenging', 'common', 'training']:
            map_gt_image = map_image.copy()
        else:
            if fast:
                map_gt_image = maps_gt_images[idx]
            else:
                map_gt_image = heat_maps_to_image(maps_gt[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)
            rgba_map_gt_image = cmap(map_gt_image)
            map_gt_image = np.delete(rgba_map_gt_image, 3, 2) * 255

        img = create_img_with_landmarks(img, img_landmarks, image_size, num_landmarks, scale=scale,
                                        circle_size=circle_size)

        merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] = img
        merged[i * image_size:(i + 1) * image_size, (j * 3 + 1) * image_size:(j * 3 + 2) * image_size, :] = map_image
        merged[i * image_size:(i + 1) * image_size, (j * 3 + 2) * image_size:(j * 3 + 3) * image_size, :] = map_gt_image

    return merged


def map_comapre_channels(images,maps1, maps2, image_size=64, num_landmarks=68, scale='255',test_data='full'):
    map1 = maps1[0]
    map2 = maps2[0]
    image = images[0]

    if image.shape[0] is not image_size:
        image = zoom(image, (0.25, 0.25, 1))
    if scale is '1':
        image *= 255
    elif scale is '0':
        image = 127.5 * (image + 1)

    row = np.ceil(np.sqrt(num_landmarks)).astype(np.int64)
    merged = np.zeros([row * image_size, row * image_size * 2, 3])

    for idx in range(num_landmarks):
        i = idx // row
        j = idx % row
        channel_map = map_to_rgb(normalize_map(map1[:, :, idx]))
        if test_data not in ['full', 'challenging', 'common', 'training']:
            channel_map2=channel_map.copy()
        else:
            channel_map2 = map_to_rgb(normalize_map(map2[:, :, idx]))

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = channel_map
        merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = channel_map2

    i = (idx + 1) // row
    j = (idx + 1) % row
    merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = image

    return merged


def train_val_shuffle_inds_per_epoch(valid_inds, train_inds, train_iter, batch_size, log_path, save_log=True):
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
