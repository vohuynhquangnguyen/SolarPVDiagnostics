import cv2
import numpy as np
import sklearn as sk
from scipy import stats


def compute_mean(image):
    return np.mean(image)


def compute_median(image):
    return np.median(image)


def compute_std(image):
    return np.std(image)


def compute_max(image):
    return np.amax(image, axis=0)


def compute_min(image):
    return np.amin(image, axis=0)


def compute_mode(image):
    return stats.mode(image, axis=0)
