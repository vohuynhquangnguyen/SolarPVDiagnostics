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
    Y_pred = np.where(Y_pred > defective_probability)
    model_confusion_matrix = confusion_matrix(Y_test, Y_pred , normalize='pred')
    
    return model_confusion_matrix

########################
# CLASSIFICATION MODEL #
########################

def compute_CAM(target_model: object, final_convolution_layer: str, 
    X_test: object):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()

    model = models.load_model(target_model)
    cam_model = models.Model(inputs = model.input, outputs = \
            (model.get_layer(final_convolution_layer).output, model.get_layer('output').output))
    features, results = cam_model.predict(X_test)
    gap_weights = model.get_layer('output').get_weights()[0]

    cams = []
    
    for idx in range(X_test.shape[0]):
        image_features = features[idx, :, :, :]
        prediction = np.argmax(results[idx])
        cam_weights = gap_weights[:, prediction]
        cam_features = \
            sp.ndimage.zoom(image_features, (X_test.shape[1] / features.shape[1], X_test.shape[1] / features.shape[2], 1), order = 5)
        cam_output = np.dot(cam_features, cam_weights)
        cams.append(cam_output)

    return np.array(cams)