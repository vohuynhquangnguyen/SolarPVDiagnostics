from cProfile import run
import os
import glob
import cv2
import numpy as np
import pandas as pd

def load_images_from_directory(images_directory: str, images_format: str) -> object:
    """
    @author: Vo Huynh Quang Nguyen
    Load all images having the same format in a directory.

    This method load_images_from_directory load all images having the same format in a directory by leveraging the Unix style pathname pattern expansion.
    
    @param images_directory: Unix style pathname. The default value is './dataset/images'
    @param images_format : Unix style user-specific image format ('*.png', '*.jpg', '*.bmp', etc.).
    @return images: Array containing loaded images.
    """

    path = os.path.join(images_directory, images_format)
    image_fnames = glob.glob(path)
    
    images = []
    for _ , image_fname in enumerate(image_fnames):
        image = cv2.imread(image_fname, cv2.IMREAD_UNCHANGED)
        images.append(image)
        
    return np.array(images)

def load_data_from_file(file_path: str, run_on_notebook: bool) -> tuple[object, object, object]:
    """
    @author: Vo, Huynh Quang Nguyen
    Load images, labels, and other features from a file. 

    This method load_data_from_file load the dataset from a .txt or .csv file containing all necessary information including <path/to/data_points>, <data_points_class>, and <other/features>.

    @param file_path: User-specified path to file containing dataset information.
    @param run_on_notebook: Whether this method is run on a Jupyter Notebook (i.e., running this method in an interactive shell).
    @return cell_images: Array containing loaded images.
    @return cell_quality: Array containing corresponding class.
    @return cell_types: Array containing corresponding type.
    """
    metadata = pd.read_csv(file_path, delim_whitespace = True, header = None)
    metadata.columns = ['cell_path', 'cell_quality', 'cell_types']

    cell_images = []
    for cell_path in metadata['cell_path']:
        if run_on_notebook == True:
            cell_path = os.path.join('../data', cell_path)
        else:
            cell_path = os.path.join('data', cell_path)
        image = cv2.imread(cell_path, cv2.IMREAD_UNCHANGED)
        cell_images.append(image)

    metadata.cell_quality = metadata['cell_quality'].map(lambda value: 'bad' if value > 0 else 'good')

    return np.array(cell_images), metadata['cell_quality'].to_numpy(), metadata['cell_types'].to_numpy()

def preprocess_data(data: object, labels: object, types: object) -> tuple[object, object, 
    object]:
    """
    @author: Vo, Huynh Quang Nguyen
    """

    preprocessed_data = np.stack((data,) * 3, axis = -1)
    preprocessed_labels = np.where(labels == 'good', 0.0, 1.0)

    idx = np.random.permutation(preprocessed_data.shape[0])
    X, Y, Z = preprocessed_data[idx], preprocessed_labels[idx], types[idx]
    
    return X, Y, Z
