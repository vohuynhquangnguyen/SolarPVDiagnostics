import numpy as np
import scipy as sp
from keras import models
import keras.backend as K
from sklearn.metrics import roc_curve, auc, confusion_matrix

###########
# METHODS #
###########
def compute_F1_score(precision: float, recall: float):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    F1_score = 1.0 /  (1.0 / precision + 1.0 / recall)
    
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

def compute_CAM(target_model: object, target_image: object, final_convolution_layer: str, feature_dim_1, feature_dim_2):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()

    model = models.load_model(target_model)
    cam_model = models.Model(inputs = model.input, outputs = \
            (model.get_layer(final_convolution_layer).output, model.layers[-1].output))
    features, results = cam_model.predict(target_image[None])
    gap_weights = model.layers[-1].get_weights()[0]

    prediction = np.argmax(results)
    cam_weights = gap_weights[:, prediction]
    cam_features = sp.ndimage.zoom(features[0], (300 / feature_dim_1, 300 / feature_dim_2, 1), order = 2)
    print(cam_features.shape)
    cam_output = np.dot(cam_features, cam_weights)
    print(cam_output.shape)

    return cam_output, prediction