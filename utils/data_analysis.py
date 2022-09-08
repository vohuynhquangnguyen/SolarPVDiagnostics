import cv2
import numpy as np
import sklearn as sk
from scipy import stats


def compute_mean(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return np.mean(image.flatten())


def compute_median(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return np.median(image.flatten())


def compute_std(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return np.std(image.flatten())


def compute_max(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return np.amax(image.flatten(), axis=0)


def compute_min(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return np.amin(image.flatten(), axis=0)


def compute_mode(image: np.ndarray) -> float:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    return stats.mode(image.flatten(), axis=0)[0]


def compute_means_from_images(images):
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    means = []
    for image in images:
        mean = compute_mean(image)
        means.append(mean)
    return np.array(means)


def compute_median_from_images(images):
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    medians = []
    for image in images:
        median = compute_median(image)
        medians.append(median)
    return np.array(medians)


def compute_max_from_images(images):
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    maxs = []
    for image in images:
        max_value = compute_max(image)
        maxs.append(max_value)
    return np.array(maxs)


def compute_min_from_images(images):
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    mins = []
    for image in images:
        min_value = compute_min(image)
        mins.append(min_value)
    return np.array(mins)


def compute_mode_from_images(images):
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh
    """
    modes = []
    for image in images:
        mode = compute_mode(image)
        modes.append(mode)
    return np.appray(modes)

