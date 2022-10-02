import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def compute_image_embeddings_PCA(images: object, labels: object, types: object, n_of_components: int) \
    -> tuple[object, list]:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    flatten_images = np.array([image.ravel() for image in images])
    pca = PCA(n_of_components) 
    embeddings = pca.fit_transform(flatten_images.data)
    
    embedding_classes = []
    for label, type in zip(list(labels), list(types)):
        if label == 0.0 and type == 'mono': # 'mono-functional'
            embedding_class = 1
        elif (label == 0.3333333333333333 or label == 0.6666666666666666) \
            and type == 'mono': # 'mono-marginally-defective'
            embedding_class = 2
        elif label == 1.0 and type == 'mono': # 'mono-defective'
            embedding_class = 3
        elif label == 0.0 and type == 'poly': # 'poly-functional'
            embedding_class = 4
        elif (label == 0.3333333333333333 or label == 0.6666666666666666) \
            and type == 'poly': # 'poly-marginally-defective'
            embedding_class = 5
        elif label == 1.0 and type == 'poly': # 'poly-defective'
            embedding_class = 6
        embedding_classes.append(embedding_class)

    return embeddings, embedding_classes

def compute_image_embeddings_tSNE(images: object, labels: object, types: object) -> tuple[object, list]:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    flatten_images = np.array([image.ravel() for image in images])
    tsne = TSNE(n_components = 2, random_state = 0)
    embeddings = tsne.fit_transform(flatten_images)

    embedding_classes = []
    for label, type in zip(list(labels), list(types)):
        if label == 0.0 and type == 'mono': # 'mono-functional'
            embedding_class = 1
        elif (label == 0.3333333333333333 or label == 0.6666666666666666) \
            and type == 'mono': # 'mono-marginally-defective'
            embedding_class = 2
        elif label == 1.0 and type == 'mono': # 'mono-defective'
            embedding_class = 3
        elif label == 0.0 and type == 'poly': # 'poly-functional'
            embedding_class = 4
        elif (label == 0.3333333333333333 or label == 0.6666666666666666) \
            and type == 'poly': # 'poly-marginally-defective'
            embedding_class = 5
        elif label == 1.0 and type == 'poly': # 'poly-defective'
            embedding_class = 6
        embedding_classes.append(embedding_class)

    return embeddings, embedding_classes

