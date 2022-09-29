import time
from tkinter import Image
import tensorflow as tf
import numpy as np
from keras import Model, layers, regularizers, optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.applications import VGG19, ResNet152V2, InceptionResNetV2, EfficientNetB7
import keras.backend as K

###########
# METHODS #
###########
def configure_training_policy():
    """
    @author: Vo, Huynh Quang Nguyen

    Configure TensorFlow and Keras training policy.
    """
    physical_devices  =  tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32 = True)

            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as error:
            print(error)

    return None

def train_classification_model(model: object, model_name: str, version: str, X: object, Y: object, 
    metric_to_monitor: str, no_of_epochs: int, batch_size: int, validation_split_ratio: float) -> tuple[object, float]:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()
    start_time = time.time()
    ###
    weight_path = f'weights/{model_name}_{version}.hdf5'
    checkpoint = ModelCheckpoint(weight_path, monitor = metric_to_monitor, 
        verbose = 1, save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint]
    history = model.fit(X, Y, validation_split = validation_split_ratio, epochs = no_of_epochs, 
        batch_size = batch_size, callbacks = callbacks_list, verbose = 1)
    np.save(f'{model_name}_history.npy', history.history)
    ###
    end_time = time.time()

    training_time = round(end_time - start_time, 4)
    return history, training_time


#####################
# SUPPORTING LAYERS #
#####################
def normalize_and_augmentation(input_tensor: object) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    rescaling = layers.Rescaling(1./255)(input_tensor)
    flipping = layers.RandomFlip()(rescaling)
    rotating = layers.RandomRotation(factor = (-0.5,0.5), 
        fill_mode = 'nearest', interpolation = 'bilinear')(flipping)
    zooming = layers.RandomZoom(height_factor = 0.02, width_factor = 0.02)(rotating)
    output_tensor = layers.RandomTranslation(height_factor = 0.02, width_factor = 0.02, fill_mode = 'nearest', interpolation = 'bilinear')(zooming)

    return output_tensor

########################
# CLASSIFICATION MODEL #
########################
def vgg19(input_shape: tuple, display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized VGG19 binary classification model.

    This method vgg19 creates a transfer-learning customized VGG19 binary classification model. If prompted by users, the model's information will be printed on the display.

    @param input_shape. Dimension of input data in the format of (height, width, channels). Minimum supported dimension is (32, 32, 3).
    @param display_model_information. Whether to display model's information.
    """

    K.clear_session()
    inputs = layers.Input(shape = input_shape, name = 'inputs')
    normalized_augmented = normalize_and_augmentation(inputs)
    vggmodel = VGG19(include_top = False, input_tensor = normalized_augmented, weights = 'imagenet')
    vggmodel.trainable = False
    x = vggmodel.output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', kernel_initializer = 'he_normal', 
        name = 'dense1')(x)
    x = layers.Dense(4096, activation = 'relu', kernel_initializer = 'he_normal',
        name = 'dense2')(x)    
    x = layers.Dense(1000, activation = 'relu', kernel_initializer = 'he_normal',
        name = 'dense3')(x)    
    x = layers.Dense(512, activation = 'relu', kernel_initializer = 'he_normal',
         name = 'dense4')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'he_normal', 
        name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'VGG19')
    
    model.compile(loss = 'binary_crossentropy', 
        optimizer = optimizers.Adam(learning_rate = 0.0001), 
        metrics = ['accuracy', 'Precision', 'Recall'])
    
    if (display_model_information == True):
        model.summary()

    return model

def resnet152v2(input_shape: tuple, display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized ResNet152v2 binary classification model.

    This method resnet152v2 creates a transfer-learning customized ResNet152v2 binary classification model. If prompted by users, the model's information will be printed on the display.

    @param input_shape. Dimension of input data in the format of (height, width, channels). Minimum supported dimension is (32, 32, 3).
    @param display_model_information. Whether to display model's information.
    """
    
    K.clear_session()
    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    convolutional_base = ResNet152V2(include_top = False, input_tensor = normalized_augmented,
        weights = 'imagenet')
    convolutional_base.trainable = False
    x = convolutional_base.get_layer(-1).output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dense(1000, activation = 'relu', name = 'dense3')(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense4')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'ResNet152v2')
    
    model.compile(loss = 'binary_crossentropy', 
        optimizer = optimizers.Adam(learning_rate = 0.0001), 
        metrics = ['accuracy', 'Precision', 'Recall'])

    if (display_model_information == True):
        model.summary()

    return model

def inception_resnetv2(input_shape: tuple, display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized InceptionResNetv2 binary classification model.

    This method inception_resnetv2 creates a transfer-learning customized InceptionResNetv2 binary classification model. If prompted by users, the model's information will be printed on the display.

    @param input_shape. Dimension of input data in the format of (height, width, channels). Minimum supported dimension is (75, 75, 3).
    @param display_model_information. Whether to display model's information.

    """
    
    K.clear_session()
    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    convolutional_base = InceptionResNetV2(include_top = False, 
        input_tensor = normalized_augmented, weights = 'imagenet')
    convolutional_base.trainable = False
    x = convolutional_base.get_layer(-1).output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dense(1000, activation = 'relu', name = 'dense3')(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense4')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'InceptionResNetv2')
    
    if (display_model_information == True):
        model.summary()

    return model

def efficientnetB7(input_shape: tuple, display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    K.clear_session()

    inputs = layers.Input(shape = input_shape)
    convolutional_base = EfficientNetB7(include_top = False, input_tensor = inputs, weights = 'imagenet')
    convolutional_base.trainable = True
    x = convolutional_base.get_layer(-1).output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dense(1000, activation = 'relu', name = 'dense3')(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense4')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    if (display_model_information == True):
        model.summary()
    return model

########################
# RECONSTRUCTION MODEL #
########################