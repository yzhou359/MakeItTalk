from menpo_functions import *
from data_loading_functions import *
from menpo.shape import bounding_box
from menpo.transform import Translation, Rotation


def augment_face_image(img, image_size=256, crop_size=248, angle_range=30, flip=True, warp_mode='constant'):
    """basic image augmentation: random crop, rotation and horizontal flip"""

    #from menpo
    def round_image_shape(shape, round):
        if round not in ['ceil', 'round', 'floor']:
            raise ValueError('round must be either ceil, round or floor')
        # Ensure that the '+' operator means concatenate tuples
        return tuple(getattr(np, round)(shape).astype(np.int))

    # taken from MDM
    def mirror_landmarks_68(lms, im_size):
        return PointCloud(abs(np.array([0, im_size[1]]) - lms.as_vector(
        ).reshape(-1, 2))[mirrored_parts_68])

    # taken from MDM
    def mirror_image(im):
        im = im.copy()
        im.pixels = im.pixels[..., ::-1].copy()

        for group in im.landmarks:
            lms = im.landmarks[group]
            if lms.points.shape[0] == 68:
                im.landmarks[group] = mirror_landmarks_68(lms, im.shape)

        return im

    flip_rand = np.random.random() > 0.5
    #     rot_rand = np.random.random() > 0.5
    #     crop_rand = np.random.random() > 0.5
    rot_rand = True  # like ECT
    crop_rand = True  # like ECT

    if crop_rand:
        lim = image_size - crop_size
        min_crop_inds = np.random.randint(0, lim, 2)
        max_crop_inds = min_crop_inds + crop_size
        img = img.crop(min_crop_inds, max_crop_inds)

    if flip and flip_rand:
        img = mirror_image(img)

    if rot_rand:
        rot_angle = 2 * angle_range * np.random.random_sample() - angle_range
        # img = img.rotate_ccw_about_centre(rot_angle)

        # Get image's bounding box coordinates
        bbox = bounding_box((0, 0), [img.shape[0] - 1, img.shape[1] - 1])
        # Translate to origin and rotate counter-clockwise
        trans = Translation(-img.centre(),
                            skip_checks=True).compose_before(
            Rotation.init_from_2d_ccw_angle(rot_angle, degrees=True))
        rotated_bbox = trans.apply(bbox)
        # Create new translation so that min bbox values go to 0
        t = Translation(-rotated_bbox.bounds()[0])
        trans.compose_before_inplace(t)
        rotated_bbox = trans.apply(bbox)
        # Output image's shape is the range of the rotated bounding box
        # while respecting the users rounding preference.
        shape = round_image_shape(rotated_bbox.range() + 1, 'round')

        img = img.warp_to_shape(
            shape, trans.pseudoinverse(), warp_landmarks=True, mode=warp_mode)

    img = img.resize([image_size, image_size])

    return img


def augment_menpo_img_ns(img, img_dir_ns, p_ns=0, ns_ind=None):
    """texture style image augmentation using stylized copies in *img_dir_ns*"""

    if p_ns > 0.5:
        ns_augs = glob(os.path.join(img_dir_ns, img.path.name.split('.')[0] + '_ns*'))
        num_augs = len(ns_augs)
        if num_augs > 0:
            if ns_ind is None or ns_ind >= num_augs:
                ns_ind = np.random.randint(0, num_augs)
            ns_aug = mio.import_image(ns_augs[ns_ind])
            img.pixels = ns_aug.pixels
    return img


def augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=0, ns_ind=None):
    """texture style image augmentation using stylized copies in *img_dir_ns*"""

    if p_ns > 0.5:
        ns_augs = glob(os.path.join(img_dir_ns, img.path.name.split('.')[0] + '_ns*'))
        num_augs = len(ns_augs)
        if num_augs > 0:
            if ns_ind is None or ns_ind >= num_augs:
                ns_ind = np.random.randint(0, num_augs)
            # ns_aug = mio.import_image(ns_augs[ns_ind])
            # ns_pixels = ns_aug.pixels
    return img


def augment_menpo_img_geom_dont_apply(img, p_geom=0):
    """geometric style image augmentation using random face deformations"""

    if p_geom > 0.5:
        lms_geom_warp = deform_face_geometric_style(img.landmarks['PTS'].points.copy(), p_scale=p_geom, p_shift=p_geom)
    return img


def augment_menpo_img_geom(img, p_geom=0):
    """geometric style image augmentation using random face deformations"""

    if p_geom > 0.5:
        lms_geom_warp = deform_face_geometric_style(img.landmarks['PTS'].points.copy(), p_scale=p_geom, p_shift=p_geom)
        img=warp_face_image_tps(img, PointCloud(lms_geom_warp))
    return img


def load_menpo_image_list(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_geom(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom_dont_apply(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_texture(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_artistic(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom_dont_apply(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list