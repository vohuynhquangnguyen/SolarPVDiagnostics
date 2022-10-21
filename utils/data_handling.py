import os
import glob
import cv2
import numpy as np
import pandas as pd

def load_data_from_file(file_path: str):
    """
    @author: Vo, Huynh Quang Nguyen
    
    Load data, labels, and types from a file. 

    This `load_data_from_file` method loads the ELPV dataset provided by Buerhop et al. from a .txt or .csv file containing all necessary information including `<path/to/data_points>`, `<data_points_class>`, and `<data_points_types>`.

    @param `file_path`: User-specified path to file containing dataset information.
    @return `cell_images`: Array containing  cell images.
    @return `cell_quality`: Array containing cell defective probabilities.
    @return `cell_types`: Array containing cell types.
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

    This `query_data_by_labels_and_types` method queries the dataset by user-specified labels and types to get the corresponding filtered data points.

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

def preprocess_data(data: object, labels: object, types: object):
    """
    @author: Vo, Huynh Quang Nguyen

    Preprocess the ELPV data provided by Buerhop et al. for classification models.

    This `preprocess_data` method applies several preprocessing methods to the ELPV dataset such that:
    1. All grayscale images are transformed into pseudo-RGB images.
    2. All image classes are one-hot encoded (0.0 vs 1.0, respectively).
    3. All data points are randomly shuffled if prompted.

    @param `data`: Array containing cell images.
    @param `labels`: Array containing cell defective probabilities.
    @param `types`: Array containing cell types.
    @return `X`: Array containing preprocessed cell images.
    @return `Y`: Array containing encoded cell defective probabilities.
    @return `Z`: Array containing preprocessed cell types.    
    """

    preprocessed_data = np.stack((data,) * 3, axis = -1)
    preprocessed_labels = labels.copy()
    preprocessed_labels[preprocessed_labels == 0.3333333333333333] = 1.0
    preprocessed_labels[preprocessed_labels == 0.6666666666666666] = 1.0

    idx = np.random.permutation(preprocessed_data.shape[0])
    X, Y, Z = preprocessed_data[idx], preprocessed_labels[idx], types[idx]
    
    return X, Y, Z
