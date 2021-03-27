from thirdparty.face_of_art.logging_functions import *
import os
import numpy as np
from menpo.shape import PointCloud
from menpofit.clm import GradientDescentCLMFitter
import pickle
import math

jaw_line_inds = np.arange(0, 17)
nose_inds = np.arange(27, 36)
left_eye_inds = np.arange(36, 42)
right_eye_inds = np.arange(42, 48)
left_brow_inds = np.arange(17, 22)
right_brow_inds = np.arange(22, 27)
mouth_inds = np.arange(48, 68)


def sigmoid(x, rate, offset):
    return 1 / (1 + math.exp(-rate * (x - offset)))


def calculate_evidence(patch_responses, rate=0.25, offset=20):
    # from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment

    rspmapShape = patch_responses[0, 0, ...].shape
    n_points = patch_responses.shape[0]

    y_weight = [np.sum(patch_responses[i, 0, ...], axis=1) for i in range(n_points)]
    x_weight = [np.sum(patch_responses[i, 0, ...], axis=0) for i in range(n_points)]

    # y_weight /= y_weight.sum()
    # x_weight /= x_weight.sum()

    y_coordinate = range(0, rspmapShape[0])
    x_coordinate = range(0, rspmapShape[1])

    varList = [(np.abs(
        np.average((y_coordinate - np.average(y_coordinate, weights=y_weight[i])) ** 2, weights=y_weight[i])),
                np.abs(np.average((x_coordinate - np.average(x_coordinate, weights=x_weight[i])) ** 2,
                                  weights=x_weight[i])))
               for i in range(n_points)]

    # patch_responses[patch_responses<0.001] = 0
    prpList = [
        (np.sum(patch_responses[i, 0, ...], axis=(-1, -2)), np.sum(patch_responses[i, 0, ...], axis=(-1, -2)))
        for i in range(n_points)]

    var = np.array(varList).flatten()
    var[var == 0] = np.finfo(float).eps
    var = np.sqrt(var)
    var = 1 / var

    weight = np.array(prpList).flatten()
    weight *= var

    # offset = np.average(weight) - 20
    weight = [sigmoid(i, rate, offset) for i in weight]

    weight = np.array(weight)

    return weight


def get_patches_around_landmarks(heat_maps, menpo_shape, patch_size=(30,30), image_shape=256):
    # from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment

    padH = int(image_shape / 2)
    padW = int(image_shape / 2)

    rps_zeros = np.zeros((1, 2 * image_shape, 2 * image_shape, menpo_shape.n_points))
    rps_zeros[0, padH:padH + image_shape, padW:padW + image_shape, :] = heat_maps

    rOffset = np.floor(patch_size[0] / 2).astype(int)
    lOffset = patch_size[0] - rOffset

    rspList = [rps_zeros[0, y - rOffset:y + lOffset, x - rOffset:x + lOffset, i] for i in range(menpo_shape.n_points)
               for y in [np.around(menpo_shape.points[i][0] + 1 + padH).astype(int)]
               for x in [np.around(menpo_shape.points[i][1] + 1 + padW).astype(int)]]
    patches = np.array(rspList)[:, None, :, :]
    return patches


def pdm_correct(init_shape, pdm_model, part_inds=None):
    """ correct landmarks using pdm (point distribution model)"""
    pdm_model.set_target(PointCloud(init_shape))
    if part_inds is None:
        return pdm_model.target.points
    else:
        return pdm_model.target.points[part_inds]


def weighted_pdm_transform(input_pdm_model, patches, shape, inirho=20):
    # from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment
    weight = calculate_evidence(patches, rate=0.5, offset=10).reshape((1, -1))
    pdm_model = input_pdm_model.copy()

    # write project_weight
    ini_rho2_inv_prior = np.hstack((np.zeros((4,)), inirho / pdm_model.model.eigenvalues))
    J = np.rollaxis(pdm_model.d_dp(None), -1, 1)
    J = J.reshape((-1, J.shape[-1]))

    initial_shape_mean = shape.points.ravel() - pdm_model.model._mean
    iniJe = - J.T.dot(initial_shape_mean * weight[0])
    iniJWJ = J.T.dot(np.diag(weight[0]).dot(J))
    inv_JJ = np.linalg.inv(iniJWJ + np.diag(ini_rho2_inv_prior))
    initial_p = -inv_JJ.dot(iniJe)

    # Update pdm
    pdm_model._from_vector_inplace(initial_p)
    return pdm_model.target.points


def w_pdm_correct(init_shape, patches, pdm_model, part_inds=None):
    """ correct landmarks using weighted pdm"""

    points = weighted_pdm_transform(input_pdm_model=pdm_model, patches=patches, shape=PointCloud(init_shape))

    if (part_inds is not None and pdm_model.n_points < 68) or part_inds is None:
        return points
    else:
        return points[part_inds]


def feature_based_pdm_corr(lms_init, models_dir, train_type='basic', patches=None):
    """ correct landmarks using part-based pdm"""

    jaw_line_inds = np.arange(0, 17)
    nose_inds = np.arange(27, 36)
    left_eye_inds = np.arange(36, 42)
    right_eye_inds = np.arange(42, 48)
    left_brow_inds = np.arange(17, 22)
    right_brow_inds = np.arange(22, 27)
    mouth_inds = np.arange(48, 68)

    '''
    selected number of PCs:
    jaw:7
    eye:3
    brow:2
    nose:5
    mouth:7
    '''

    new_lms = np.zeros((68, 2))

    parts = ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'mouth', 'nose', 'jaw']
    part_inds_opt = [left_brow_inds, right_brow_inds, left_eye_inds, right_eye_inds, mouth_inds, nose_inds,
                     jaw_line_inds]
    pc_opt = [2, 2, 3, 3, 7, 5, 7]

    for i, part in enumerate(parts):
        part_inds = part_inds_opt[i]
        pc = pc_opt[i]
        temp_model = os.path.join(models_dir, train_type + '_' + part + '_' + str(pc))
        filehandler = open(temp_model, "rb")
        try:
            pdm_temp = pickle.load(filehandler)
        except UnicodeDecodeError:
            pdm_temp = pickle.load(filehandler, fix_imports=True, encoding="latin1")
        filehandler.close()

        if patches is None:
            part_lms_pdm = pdm_correct(lms_init[part_inds], pdm_temp)
        else:
            part_lms_pdm = w_pdm_correct(
                init_shape=lms_init[part_inds], patches=patches, pdm_model=pdm_temp, part_inds=part_inds)

        new_lms[part_inds] = part_lms_pdm
    return new_lms


def clm_correct(clm_model_path, image, map, lms_init):
    """ tune landmarks using clm (constrained local model)"""

    filehandler = open(os.path.join(clm_model_path), "rb")
    try:
        part_model = pickle.load(filehandler)
    except UnicodeDecodeError:
        part_model = pickle.load(filehandler, fix_imports=True, encoding="latin1")
    filehandler.close()

    # from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment
    part_model.opt = dict()
    part_model.opt['numIter'] = 5
    part_model.opt['kernel_covariance'] = 10
    part_model.opt['sigOffset'] = 25
    part_model.opt['sigRate'] = 0.25
    part_model.opt['pdm_rho'] = 20
    part_model.opt['verbose'] = False
    part_model.opt['rho2'] = 20
    part_model.opt['ablation'] = (True, True)
    part_model.opt['ratio1'] = 0.12
    part_model.opt['ratio2'] = 0.08
    part_model.opt['smooth'] = True

    fitter = GradientDescentCLMFitter(part_model, n_shape=30)

    image.rspmap_data = np.swapaxes(np.swapaxes(map, 1, 3), 2, 3)

    fr = fitter.fit_from_shape(image=image, initial_shape=PointCloud(lms_init), gt_shape=PointCloud(lms_init))
    w_pdm_clm = fr.final_shape.points

    return w_pdm_clm
