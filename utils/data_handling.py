import os
import glob
import cv2
import numpy as np

def load_images_from_directory(images_directory: str, images_format: str):
    """
    @author: Vo Huynh Quang Nguyen
    Load all images having the same format in a directory.

    This function load_images_from_directory load all images having the same format in a directory by leveraging the Unix style pathname pattern expansion.
    
    @param images_directory: Unix style pathname. The default value is './dataset/images'
    @param images_format : Unix style user-specific image format. The default value is ''.
        
    Returns
    ----------
    images : np.ndarray with shape (data points, height, width, channel)
        Array containing images.
    """
    path = os.path.join(directory, image_format)
    image_fnames = glob.glob(path)
    
    images = []
    for _ , image_fname in enumerate(image_fnames):
        image = cv2.imread(image_fname, cv2.IMREAD_UNCHANGED)
        images.append(image)
        
    return np.array(images)