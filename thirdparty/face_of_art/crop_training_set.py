from scipy.misc import imsave
from menpo_functions import *
from data_loading_functions import *


# define paths & parameters for cropping dataset
img_dir = '~/landmark_detection_datasets/'
dataset = 'training'
bb_type = 'gt'
margin = 0.25
image_size = 256

# load bounding boxes
bb_dir = os.path.join(img_dir, 'Bounding_Boxes')
bb_dictionary = load_bb_dictionary(bb_dir, mode='TRAIN', test_data=dataset)

# directory for saving face crops
outdir = os.path.join(img_dir, 'crop_'+bb_type+'_margin_'+str(margin))
if not os.path.exists(outdir):
    os.mkdir(outdir)

# load images
imgs_to_crop = load_menpo_image_list(
    img_dir=img_dir, train_crop_dir=None, img_dir_ns=None, mode='TRAIN', bb_dictionary=bb_dictionary,
    image_size=image_size, margin=margin, bb_type=bb_type, augment_basic=False)

# save cropped images with matching landmarks
print ("\ncropping dataset from: "+os.path.join(img_dir, dataset))
print ("\nsaving cropped dataset to: "+outdir)
for im in imgs_to_crop:
    if im.pixels.shape[0] == 1:
        im_pixels = gray2rgb(np.squeeze(im.pixels))
    else:
        im_pixels = np.rollaxis(im.pixels, 0, 3)
    imsave(os.path.join(outdir, im.path.name.split('.')[0]+'.png'), im_pixels)
    mio.export_landmark_file(im.landmarks['PTS'], os.path.join(outdir, im.path.name.split('.')[0]+'.pts'))

print ("\ncropping dataset completed!")
