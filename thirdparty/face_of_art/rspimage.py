# copied from ECT: https://github.com/HongwenZhang/ECT-FaceAlignment

import numpy as np
import math
import scipy

from menpo.image import Image
from menpo.shape.pointcloud import PointCloud


def sigmoid(x, rate, offset):
    return 1 / (1 + math.exp(-rate*(x-offset)))


def initial_shape_fromMap(image):
    # a = np.random.random((10, 10))
    rspmapShape = image.rspmap_data[0, 0,...].shape
    n_points = image.rspmap_data.shape[1]
    pointsData = np.array([np.unravel_index(image.rspmap_data[0, i,...].argmax(), rspmapShape) for i in range(n_points)], dtype=np.float32)
    # width_ratio = float(image.shape[1])/image.rspmap_data.shape[3]
    # height_ratio = float(image.shape[0])/ image.rspmap_data.shape[2]
    # pointsData *= [height_ratio, width_ratio]
    points = PointCloud(pointsData)
    points.project_weight = None

    return points


def calculate_evidence(patch_responses, rate=0.25, offset=20):
    rspmapShape = patch_responses[0, 0,...].shape
    n_points = patch_responses.shape[0]

    y_weight = [np.sum(patch_responses[i,0,...], axis=1) for i in range(n_points)]
    x_weight = [np.sum(patch_responses[i,0,...], axis=0) for i in range(n_points)]

    # y_weight /= y_weight.sum()
    # x_weight /= x_weight.sum()

    y_coordinate = range(0, rspmapShape[0])
    x_coordinate = range(0, rspmapShape[1])

    varList = [(np.abs(np.average((y_coordinate - np.average(y_coordinate, weights=y_weight[i]))**2, weights=y_weight[i])),
                np.abs(np.average((x_coordinate - np.average(x_coordinate, weights=x_weight[i])) ** 2, weights=x_weight[i])))
                for i in range(n_points)]

    # patch_responses[patch_responses<0.001] = 0
    prpList = [(np.sum(patch_responses[i,0,...], axis=(-1, -2)), np.sum(patch_responses[i,0,...], axis=(-1, -2)))  for i in range(n_points) ]

    var = np.array(varList).flatten()
    var[var == 0] = np.finfo(float).eps
    var = np.sqrt(var)
    var = 1/var

    weight = np.array(prpList).flatten()
    weight *= var

    # offset = np.average(weight) - 20
    weight = [sigmoid(i, rate, offset) for i in weight]

    weight = np.array(weight)

    return weight


class RspImage(Image):
    r"""
    RspImage is Image with response map
    """
    def __init__(self, image_data, rspmap_data = None):
        super(RspImage, self).__init__(image_data)
        self.rspmap_data = rspmap_data

    @classmethod
    def init_from_image(cls, image):
        image.__class__ = RspImage
        image.rspmap_data = None
        return image

    def set_rspmap(self, rspmap_data):
        self.rspmap_data = rspmap_data