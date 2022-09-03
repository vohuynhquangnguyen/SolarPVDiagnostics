import os
import glob
import cv2
import numpy as np

def load_images_from_directory(images_directory: str, images_format: str) -> object:
    """
    @author: Vo Huynh Quang Nguyen
    Load all images having the same format in a directory.

    This function load_images_from_directory load all images having the same format in a directory by leveraging the Unix style pathname pattern expansion.
    
    @param images_directory: Unix style pathname. The default value is './dataset/images'
    @param images_format : Unix style user-specific image format ('*.png', '*.jpg', '*.bmp', etc.).
    """
    path = os.path.join(images_directory, images_format)
    image_fnames = glob.glob(path)
    
    images = []
    for _ , image_fname in enumerate(image_fnames):
        image = cv2.imread(image_fname, cv2.IMREAD_UNCHANGED)
        images.append(image)
        
    return np.array(images)

def load_data_from_file(filepath: str, delimiter: str) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    Load images, labels, and other features from a file. 

    This function load_data_from_file load the dataset from a .txt or .csv file containing all necessary information including <path/to/data_points>, <data_points_class>, and <other/features>.

    @param filepath: User-specified path to file containing dataset information. 
    @param delimiter: Delimiter (e.g., ',', ' ', '\t', '\v', etc.).
    """

    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        n_columns = len(lines[0].split(delimiter))

        if (n_columns == 3):
            images = []
            labels = []
            cells_type = []

            for line in lines:
                path, label, type = line.split(delimiter)
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                images.append(image)
                labels.append(label)
                cells_type.append(type)
            return images, labels, cells_type
        else:
            pass