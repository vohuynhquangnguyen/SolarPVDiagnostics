import cv2
import numpy as np
import sklearn as sk
from scipy import stats

def compute_statistical_parameters(images: object) -> tuple[]:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh

    Compute statistical parameters of an input image.
    """
    means, medians, stds, modes, maxs, mins = [], [], [], [], [], [], []

    for image in images:
        mean = np.mean(image.flatten())
        means.append(mean)

        median = np.median(image.flatten())
        medians.append(median)

        std = np.std(image.flatten())
        stds.append(std)

        amax = np.amax(image.flatten(), axis = 0)
        maxs.append(amax)

        amin = np.amin(image.flatten(), axis = 0)
        mins.append(amin)

        mode = stats.mode(image.flatten(), axis = 0)
        modes.append(mode)

    return means, medians, stds, maxs, mins, modes

def average_image(images):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    avg_image = images[0]
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)
    return avg_image
