import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

def compute_statistical_parameters(images: object) -> tuple[object, object, object, object, object, object]:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh

    Compute statistical parameters image-by-image from a given dataset.
    """

    means = np.array([np.mean(image.ravel()) for image in images])
    median = np.array([np.median(image.ravel()) for image in images])
    stds = np.array([np.std(image.ravel()) for image in images])
    maxs = np.array([np.amax(image.ravel(), axis = 0) for image in images])
    mins = np.array([np.amin(image.ravel(), axis = 0) for image in images])
    modes = np.array([stats.mode(image.ravel(), axis = 0) for image in images])

    return means, median, stds, maxs, mins, modes

def compute_average_image(images: object) -> object:
    """
    @author: Vo, Huynh Quang Nguyen; Hoang, Minh

    Compute an average image from a given dataset.
    """
    average_image = np.zeros(images[0].shape)

    for image in images:
        average_image = np.add(average_image, image)
    
    average_image /= images.shape[0]

    return average_image

def compute_image_embedding(images: object, n_of_components: int):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    flatten_images = np.array([image.ravel() for image in images])
    pca = PCA(n_of_components) 
    transformed_images = pca.fit_transform(flatten_images.data)
    
    return transformed_images

def compute_eigenimages(images: object, explained_variance_ratio: float):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    assert (explained_variance_ratio >= 0.0) and (explained_variance_ratio <= 1.0), \
        print('The ratio must higher than 0.0 and lower than 1.1')

    flatten_images = np.array([image.ravel() for image in images])
    pca = PCA(explained_variance_ratio) 
    transformed_images = pca.fit_transform(flatten_images.data)

    return transformed_images

