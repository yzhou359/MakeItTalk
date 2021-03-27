from create_art_data_functions import *
from scipy.misc import imsave
import sys


'''THIS SCRIPT CREATES PRE-AUGMENTED DATA TO SAVE TRAINING TIME (ARTISTIC OR BASIC AUGMENTATION):
    under the folder *outdir*, it will create a separate folder for each epoch. the folder will
    contain the augmented images and matching landmark (pts) files.'''

# parameter for calculating number of epochs
num_train_images = 3148  # number of training images
train_iter = 100000  # number of training iterations
batch_size = 6  # batch size in training
num_epochs = int(np.ceil((1. * train_iter) / (1. * num_train_images / batch_size)))+1

# augmentation parameters
num_augs = 9  # number of style transfer augmented images
aug_geom = True  # use artistic geometric augmentation?
aug_texture = True  # use artistic texture augmentation?

# image parameters
bb_type = 'gt'  # face bounding-box type (gt/init)
margin = 0.25  # margin for face crops - % of bb size
image_size = 256  # image size

# data-sets image paths
dataset = 'training'  # dataset to augment (training/full/common/challenging/test)
img_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'
train_crop_dir = 'crop_gt_margin_0.25'  # directory of train images cropped to bb (+margin)
img_dir_ns = os.path.join(img_dir, train_crop_dir+'_ns')  # dir of train imgs cropped to bb + style transfer
outdir = '/Users/arik/Desktop/epoch_data'  # directory for saving augmented data

# other parameters
min_epoch_to_save = 0  # start saving images from this epoch (first epoch is 0)
debug_data_size = 15
debug = False
random_seed = 1234  # random seed for numpy

########################################################################################
if aug_texture and img_dir_ns is None:
    print('\n *** ERROR: aug_texture is True, and img_dir_ns is None.\n'
          'please specify path for img_dir_ns to augment image texture!')
    sys.exit()

if not os.path.exists(outdir):
    os.mkdir(outdir)

gt = (bb_type == 'gt')
bb_dir = os.path.join(img_dir, 'Bounding_Boxes')

if dataset == 'training':
    mode = 'TRAIN'
else:
    mode = 'TEST'
bb_dictionary = load_bb_dictionary(bb_dir, mode=mode, test_data=dataset)

aug_geom_dir = os.path.join(outdir, 'aug_geom')
aug_texture_dir = os.path.join(outdir, 'aug_texture')
aug_geom_texture_dir = os.path.join(outdir, 'aug_geom_texture')
aug_basic_dir = os.path.join(outdir, 'aug_basic')

if not aug_geom and aug_texture:
    save_aug_path = aug_texture_dir
elif aug_geom and not aug_texture:
    save_aug_path = aug_geom_dir
elif aug_geom and aug_texture:
    save_aug_path = aug_geom_texture_dir
else:
    save_aug_path = aug_basic_dir

print ('saving augmented images: aug_geom=' + str(aug_geom) + ' aug_texture=' + str(aug_texture) +
       ' : ' + str(save_aug_path))

if not os.path.exists(save_aug_path):
    os.mkdir(save_aug_path)

np.random.seed(random_seed)
ns_inds = np.arange(num_augs)

for i in range(num_epochs):
    print ('saving augmented images of epoch %d/%d' % (i, num_epochs-1))
    if not os.path.exists(os.path.join(save_aug_path, str(i))) and i > min_epoch_to_save - 1:
        os.mkdir(os.path.join(save_aug_path, str(i)))

    if i % num_augs == 0:
        np.random.shuffle(ns_inds)

    if not aug_geom and aug_texture:
        img_list = load_menpo_image_list_no_geom(
            img_dir=img_dir, train_crop_dir=train_crop_dir, img_dir_ns=img_dir_ns, mode='TRAIN',
            bb_dictionary=bb_dictionary,
            image_size=image_size, margin=margin, bb_type=bb_type, augment_basic=True,
            augment_texture=True, p_texture=1.,
            augment_geom=True, p_geom=1., ns_ind=ns_inds[i % num_augs], dataset=dataset)
    elif aug_geom and not aug_texture:
        img_list = load_menpo_image_list_no_texture(
            img_dir=img_dir, train_crop_dir=train_crop_dir, img_dir_ns=img_dir_ns, mode='TRAIN',
            bb_dictionary=bb_dictionary,
            image_size=image_size, margin=margin, bb_type=bb_type, augment_basic=True,
            augment_texture=True, p_texture=1.,
            augment_geom=True, p_geom=1., ns_ind=ns_inds[i % num_augs], dataset=dataset)
    elif aug_geom and aug_texture:
        img_list = load_menpo_image_list(
            img_dir=img_dir, train_crop_dir=train_crop_dir, img_dir_ns=img_dir_ns, mode='TRAIN',
            bb_dictionary=bb_dictionary,
            image_size=image_size, margin=margin, bb_type=bb_type, augment_basic=True,
            augment_texture=True, p_texture=1.,
            augment_geom=True, p_geom=1., ns_ind=ns_inds[i % num_augs], dataset=dataset)
    else:
        img_list = load_menpo_image_list_no_artistic(
            img_dir=img_dir, train_crop_dir=train_crop_dir, img_dir_ns=img_dir_ns, mode='TRAIN',
            bb_dictionary=bb_dictionary,
            image_size=image_size, margin=margin, bb_type=bb_type, augment_basic=True,
            augment_texture=True, p_texture=1.,
            augment_geom=True, p_geom=1., ns_ind=ns_inds[i % num_augs], dataset=dataset)

    if debug:
        img_list = img_list[:debug_data_size]

    for im in img_list:
        im_path = os.path.join(save_aug_path, str(i), im.path.name.split('.')[0] + '.png')
        pts_path = os.path.join(save_aug_path, str(i), im.path.name.split('.')[0] + '.pts')
        if i > min_epoch_to_save - 1:
            if not os.path.exists(im_path):
                if im.pixels.shape[0] == 1:
                    im_pixels = gray2rgb(np.squeeze(im.pixels))
                else:
                    im_pixels = np.rollaxis(im.pixels, 0, 3)
                imsave(im_path, im_pixels)
            if not os.path.exists(pts_path):
                mio.export_landmark_file(im.landmarks['PTS'], pts_path, overwrite=True)
print ('DONE!')