import cv2
import numpy as np
import scipy as sp
import tensorflow as tf
from keras import models
from tensorflow.python.ops.numpy_ops import np_config
import keras.backend as K
from sklearn.metrics import roc_curve, auc, confusion_matrix

def ssim_loss(target, reference):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    target = tf.cast(target, tf.float32)
    reference = tf.cast(reference, tf.float32)
    score = 1/2 - tf.reduce_mean(tf.image.ssim_multiscale(target, reference, max_val = 255.0))/2

    return np.round(score.numpy(), 4)

def compute_F1_score(precision: float, recall: float):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    F1_score = 2 * (precision * recall) / (precision + recall)
    
    return F1_score

def compute_AUC(target_model: object, X_test: object, Y_test: object):
    """
    @author: Vo, Huynh Quang Nguyen
    """

    K.clear_session()
    model = models.load_model(target_model)

    Y_pred = model.predict(X_test).ravel()
    model_fpr, model_tpr, _ = roc_curve(Y_test, Y_pred)
    model_auc = auc(model_fpr, model_tpr) 

    return model_auc, model_fpr, model_tpr

def compute_confusion_matrix(target_model: object, X_test: object, Y_test: object,
    defective_probability: float):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()
    model = models.load_model(target_model)

    Y_pred = model.predict(X_test).ravel()
    Y_pred = np.where(Y_pred > defective_probability, 1.0, 0.0)
    model_confusion_matrix = confusion_matrix(Y_test, Y_pred)
    
    return model_confusion_matrix

########################
# CLASSIFICATION MODEL #
########################
def get_validation_accuracy_precision_recall_F1(taget_history: str):
    """
    @author: Vo, Huynh Quang Nguyen

    Extract validation scores from an epoch containing the highest validation accuracy.

    This `get_validation_accuracy_precision_recall_F1` method extracts validation scores from an epoch containing the highest validation accuracy during the model's training process.
    
    @param `taget_history`: Targeted model's history generated from the training process.
    @return `val_results`: List containing the extracted validation scores (accuracy, precision, recall, and F1-score).
    """

    history = np.load(taget_history, allow_pickle = True).item()
    val_accuracy = np.max(history['val_accuracy'])
    val_precision = history['val_precision'][np.argmax(val_accuracy)]
    val_recall = history['val_recall'][np.argmax(val_accuracy)]
    val_F1 = compute_F1_score(val_precision, val_recall)

    val_results = [val_accuracy, val_precision, val_recall, val_F1]

    return val_results

def get_test_accuracy_precision_recall_F1(target_model: str, X_test: object, Y_test: object):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    K.clear_session()
    model = models.load_model(target_model)
    _ , test_accuracy, test_precision, test_recall = model.evaluate(X_test, Y_test)
    test_F1 = compute_F1_score(test_precision, test_recall)
    test_results = [test_accuracy, test_precision, test_recall, test_F1]

    return test_results

def compute_CAM(target_model: object, target_image: object, final_convolution_layer: str, feature_dims: tuple):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()

    model = models.load_model(target_model)
    cam_model = models.Model(inputs = model.input, outputs = \
            (model.get_layer(final_convolution_layer).output, model.layers[-1].output))
    features, results = cam_model.predict(target_image[None])
    gap_weights = model.layers[-1].get_weights()[0]
    print(features.shape)

    prediction = np.argmax(results)
    cam_weights = gap_weights[:, prediction]
    cam_features = sp.ndimage.zoom(features[0], (300 / feature_dims[0], 300 / feature_dims[1], 1), order = 2)
    print(cam_features.shape)
    cam_output = np.dot(cam_features, cam_weights)
    print(cam_output.shape)

    return cam_output, results



########################
# RECONSTRUCTION MODEL #
########################
def compute_heatmap(target_model: object, target_image: object):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    K.clear_session()
    np_config.enable_numpy_behavior()

    model = models.load_model(target_model, custom_objects = {'ssim_loss': ssim_loss})
    reconstructed_image = model.predict(target_image[None])

    heatmap = np.subtract(target_image, reconstructed_image[0]) ** 2
    heatmap = cv2.applyColorMap(heatmap.astype('uint8'), cv2.COLORMAP_JET)
    heatmap = cv2.medianBlur(heatmap, 11)
    score_of_difference = ssim_loss(target_image, reconstructed_image)

    return heatmap, reconstructed_image, score_of_difference