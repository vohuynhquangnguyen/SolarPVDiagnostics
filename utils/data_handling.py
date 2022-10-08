import os
import glob
import cv2
import numpy as np
import pandas as pd

###########
# METHODS #
###########
def load_images_from_directory(images_directory: str, images_format: str) -> object:
    """
    @author: Vo Huynh Quang Nguyen
    
    Load all images having the same format in a directory.

    This method `load_images_from_directory` loads all images having the same format in a directory by leveraging the Unix style pathname pattern expansion.
    
    @param `images_directory`: Unix style pathname. Default value is `./dataset/images`
    @param `images_format` : Unix style user-specific image format (`*.png`, `*.jpg`, `*.bmp`, etc.).
    @return `images`: Array containing loaded images.
    """

    path = os.path.join(images_directory, images_format)
    image_fnames = glob.glob(path)
    
    images = []
    for _ , image_fname in enumerate(image_fnames):
        image = cv2.imread(image_fname, cv2.IMREAD_UNCHANGED)
        images.append(image)
        
    return np.array(images)

def load_data_from_file(file_path: str) -> tuple[object, object, object]:
    """
    @author: Vo, Huynh Quang Nguyen
    
    Load data, labels, and other features from a file. 

    This method `load_data_from_file` loads the dataset from a .txt or .csv file containing all necessary information including `<path/to/data_points>`, `<data_points_class>`, and `<other/features>`.

    @param `file_path`: User-specified path to file containing dataset information.
    @return `cell_images`: Array containing loaded images.
    @return `cell_quality`: Array containing corresponding class.
    @return `cell_types`: Array containing corresponding type.
    """
    metadata = pd.read_csv(file_path, delim_whitespace = True, header = None)
    metadata.columns = ['cell_path', 'cell_quality', 'cell_types']

    cell_images = []
    for cell_path in metadata['cell_path']:
        cell_path = os.path.join('../data', cell_path)
        image = cv2.imread(cell_path, cv2.IMREAD_UNCHANGED)
        cell_images.append(image)

    return np.array(cell_images), metadata['cell_quality'].to_numpy(), metadata['cell_types'].to_numpy()

def query_data_by_labels_and_types(data: object, labels: object, types: object, filter_by_labels: float, filter_by_types: str):
    """
    @author: Vo, Huynh Quang Nguyen

    Query data by labels and types

    This function `query_data_by_labels_and_types` queries the dataset by user-specified labels and types to get the corresponding filtered data points.

    @param data:
    @param labels:
    @param types:
    @param filter_by_labels:
    @param filter_by_types:
    @return filtered_data:
    """
    
    filtered_label_indexes = np.array(np.where(labels == filter_by_labels)[0])
    filtered_types_indexes = np.array(np.where(types == filter_by_types)[0])
    filtered_indexes = list(np.intersect1d(filtered_label_indexes, filtered_types_indexes))
    filtered_data = data[filtered_indexes]
    
    return filtered_data

def preprocess_data(data: object, labels: object, types: object) -> tuple[object, object, 
    object]:
    """
    @author: Vo, Huynh Quang Nguyen

    Preprocess the data for classification models.

    This method preprocess_data applys several preprocessing methods to the given dataset such that:
    1. All grayscale images are transformed into pseudo-RGB images.
    2. All image classes are one-hot encoded (0.0 vs 1.0, respectively).
    3. All data points are randomly shuffled.
    """

    preprocessed_data = np.stack((data,) * 3, axis = -1)
    preprocessed_labels = labels.copy()
    preprocessed_labels[preprocessed_labels == 0.3333333333333333] = 1.0
    preprocessed_labels[preprocessed_labels == 0.6666666666666666] = 1.0

    idx = np.random.permutation(preprocessed_data.shape[0])
    X, Y, Z = preprocessed_data[idx], preprocessed_labels[idx], types[idx]
    
    return X, Y, Z
