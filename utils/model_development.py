import time
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG19, ResNet152V2, InceptionResNetV2, NASNetLarge
from tensorflow.python.ops.numpy_ops import np_config

###########
# METHODS #
###########
def configure_training_policy():
    """
    @author: Vo, Huynh Quang Nguyen

    Configure TensorFlow and Keras training policy.

    This function `configure_training_policy` configures TensorFlow and Keras training policy by:
    1. Enable the numpy behaviours for all tf.Tensor objects;
    2. Dynamically allocate device memories according to actual training needs.
    """
    physical_devices  =  tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            np_config.enable_numpy_behavior(prefer_float32 = True)

            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as error:
            print(error)

    return None

# def clear_memory():
#     """
#     @author: Vo, Huynh Quang Nguyen

#     Release all GPU memories.

#     This function `clear_memory` releases all GPU memories by invoking and reseting currently in-use GPU identified by the `cuda` object in `Numba`.
#     """
    
#     device = cuda.get_current_device()
#     device.reset()

#     return None

def data_augmentation() -> tuple[object, object, object, object]:
    """
    @author: Vo, Huynh Quang Nguyen

    Implement data augmentation scheme on a given dataset.

    This function `data_augmentation` implements several data augmentation schemes on a given dataset including:
    1. Random horizontal and vertical flip;
    2. Random horizontal and vertical shift by +-5% of the original dimensions;
    3. Random zoon by +-5% of the original dimensions;
    4. Random rotation by 90deg.

    @params `random_flip`, `random_translation`, `random_zoom`, `random_rotation`. The `tf.keras.Model` objects that apply data augmentation on a single or a set of images.
    """
    try:
        random_flip = Sequential(layers.RandomFlip(mode = 'horizontal_and_vertical'))
        random_translation = \
            Sequential(layers.RandomTranslation(height_factor = (-0.05, 0.05), 
                width_factor = (-0.05, 0.05), fill_mode = 'nearest', interpolation = 'bilinear'))
        random_zoom = \
            Sequential(layers.RandomZoom(height_factor = (-0.05, 0.05), width_factor = (-0.05, 0.05)))
        random_rotation = \
            Sequential(layers.RandomRotation(factor = (-0.255, -0.25), 
            fill_mode = 'nearest', interpolation = 'bilinear'))
    except RuntimeError as error:
        print(error)
    
    return random_flip, random_translation, random_zoom, random_rotation

def create_optimizer(type: str) -> object:
    if type == 'nadam':
        optimizer = \
            optimizers.Nadam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.9, epsilon = 1e-8)
    elif type == 'adam':
        optimizer = \
            optimizers.Adam(learning_rate = 1e-5, beta_1 = 0.9, beta_2 = 0.9, epsilon = 1e-8)
    else:
        pass
    return optimizer

def ssim_loss(target, reference):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    target = tf.cast(target, tf.float32)
    reference = tf.cast(reference, tf.float32)
    score = 1 - tf.reduce_mean(tf.image.ssim(target, reference, max_val = 255.0))

    return score

def train_classification_model(training_phase: int, model: object, 
    optimizer: object, training_metrics: list, model_name: str, version: str, 
    X: object, Y: object, metric_to_monitor: str, no_of_epochs: int, batch_size: int, validation_split_ratio: float) -> tuple[object, float]:
    """
    @author: Vo, Huynh Quang Nguyen

    Train classification models.

    This function `train_classification_model` initializes a training session for a classification model. 

    @param `training_phase`. Designated phase of this training session. If the `training_phase = 1`, only the model's fully connected (Dense) layers are trained. If the `training_phase = 2`, all model's layers are trained.
    @param `model`. Loaded model. If the `training_phase = 1`, the loaded model is a newly created model. If the `training_phase = 2`, the loaded model is a saved model at a specific location (e.g., `./models/weights/model.hdf5`).
    @param `optimizer`. Optimization function for model training.
    @param `training_metrics`. List of metrics for model training. For classificiation tasks, the metrics are usually `['accuracy', 'Precision', 'Recall']`.
    @params `model_name`, `version`. Name of the model and its version.
    @params `X`, `Y`. Training data and labels.
    @param `metric_to_monitor`. Which metric to monitor to acquire the best model.
    @params `no_of_epochs`, `batch_size`. Total number of epochs and batch size for this training session/
    @param `validation_split_ratio`: Split ratio to divide the training set into training and validation subsets.
    @return `history`. A dictionary containing model's training history including training accuracy, validation accuracy, etc. as function of epochs.
    @return `training_time`. Total elapsed training time.
    """
    
    K.clear_session()
    assert(training_phase == 1 or training_phase == 2), print('Unsupported command!')
    try:
        if (training_phase == 1):
            model = model
        elif (training_phase == 2):
            model = load_model(model)
            for layer in model.layers:
                layer.trainable = True    

        start_time = time.time()
        ###
        model.compile(\
                loss = 'binary_crossentropy', optimizer = optimizer, metrics = training_metrics)
        weight_path = f'../models/weights/{model_name}_{version}.hdf5'
        checkpoint = ModelCheckpoint(weight_path, monitor = metric_to_monitor, 
            verbose = 1, save_best_only = True, mode = 'max')
        callbacks_list = [checkpoint]
        history = model.fit(X, Y, validation_split = validation_split_ratio, epochs = no_of_epochs, 
            batch_size = batch_size, callbacks = callbacks_list, verbose = 1)
        np.save(f'../models/history/{model_name}_{version}', history.history)
        ###
        end_time = time.time()

        training_time = round(end_time - start_time, 4)
    except RuntimeError as error:
        print(error)

    return history, training_time

def train_reconstruction_model(training_phase: int, model: object, 
    optimizer: object, training_metrics: list, model_name: str, version: str, 
    X: object, Y: object, metric_to_monitor: str, no_of_epochs: int, batch_size: int, validation_split_ratio: float) -> tuple[object, float]:
    """
    @author: Vo, Huynh Quang Nguyen

    Train reconstruction models.

    This function `train_reconstruction_model` initializes a training session for a simple reconstruction model (e.g., convolutional autoencoder or uNet). 

    @param `training_phase`. Designated phase of this training session. If the `training_phase = 1`, only the model's decoding layers are trained. If the `training_phase = 2`, all model's layers are trained.
    @param `model`. Loaded model. If the `training_phase = 1`, the loaded model is a newly created model. If the `training_phase = 2`, the loaded model is a saved model at a specific location (e.g., `./models/weights/model.hdf5`).
    @param `optimizer`. Optimization function for model training.
    @param `training_metrics`. List of metrics for model training. For reconstruction tasks, the metrics are usually `val_loss`.
    @params `model_name`, `version`. Name of the model and its version.
    @params `X`, `Y`. Training data and labels.
    @param `metric_to_monitor`. Which metric to monitor to acquire the best model.
    @params `no_of_epochs`, `batch_size`. Total number of epochs and batch size for this training session/
    @param `validation_split_ratio`: Split ratio to divide the training set into training and validation subsets.
    @return `history`. A dictionary containing model's training history including training accuracy, validation accuracy, etc. as function of epochs.
    @return `training_time`. Total elapsed training time.
    """
    
    K.clear_session()
    assert(training_phase == 1 or training_phase == 2), print('Unsupported command!')
    try:
        if (training_phase == 1):
            model = model
        elif (training_phase == 2):
            model = load_model(model, compile = False)
            for layer in model.layers:
                layer.trainable = True    

        start_time = time.time()
        ###
        model.compile(\
                loss = training_metrics, optimizer = optimizer)
        weight_path = f'../models/weights/{model_name}_{version}.hdf5'
        checkpoint = ModelCheckpoint(weight_path, monitor = metric_to_monitor, 
            verbose = 1, save_best_only = True, mode = 'min')
        callbacks_list = [checkpoint]
        history = model.fit(X, Y, validation_split = validation_split_ratio, epochs = no_of_epochs, 
            batch_size = batch_size, callbacks = callbacks_list, verbose = 1)
        np.save(f'../models/history/{model_name}_{version}', history.history)
        ###
        end_time = time.time()

        training_time = round(end_time - start_time, 4)
    except RuntimeError as error:
        print(error)

    return history, training_time

#####################
# SUPPORTING LAYERS #
#####################
def normalize_and_augmentation(input_tensor: object) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Implement preprocessing layers.

    This function `normalize_and_augmentation` implements model's preprocessing layers for training data before feeding them to the model's convolutional base. The preprocessing layers consist of:
    1. Normalization layer that normalizes a training data point to the range `[0.0, 1.0]`.
    2. Data augmentation layers that randomly flip, rotate, zoom, and shift a training data point.

    @param `input_tensor`. A `tf.Tensor` object representing input training data.
    @return `output_tensor`. A `tf.Tensor` object representing output training data that are normalized and augmented.
    """
    rescaling = layers.Rescaling(1./255)(input_tensor)
    flipping = layers.RandomFlip(mode = 'horizontal_and_vertical')(rescaling)
    rotating = \
        layers.RandomRotation(factor = (-0.255, -0.25), fill_mode = 'nearest', interpolation = 'bilinear')(flipping)
    zooming = layers.RandomZoom(height_factor = (-0.05, 0.05), width_factor = (-0.05, 0.05))(rotating)
    translation = \
        layers.RandomTranslation(height_factor = (-0.05, 0.05), width_factor = (-0.05, 0.05), 
        fill_mode = 'nearest', interpolation = 'bilinear')(zooming)
    output_tensor = translation

    return output_tensor

########################
# CLASSIFICATION MODEL #
########################
def vgg19(input_shape: tuple, weights: str, freeze_convolutional_base: bool,
    display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized VGG19 binary classification model.

    This method `vgg19` creates a transfer-learning customized VGG19 binary classification model.

    @param `input_shape`. Dimension of input data in the format of `(height, width, channels)`. Minimum supported dimension is `(32, 32, 3)`.
    @param `weights`. Pretrained model weights.
    @param `freeze_convolutional_base`. Whether to freeze the convolution base.
    @param `display_model_information`. Whether to display model's information.
    """

    K.clear_session()
    inputs = layers.Input(shape = input_shape, name = 'inputs')
    normalized_augmented = normalize_and_augmentation(inputs)
    vgg19 = VGG19(include_top = False, input_tensor = normalized_augmented, weights = weights)
    if freeze_convolutional_base == True:
        vgg19.trainable = False
    else:
        vgg19.trainable = True
    x = vgg19.output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense1')(x)
    x = layers.Dropout(0.2, name = 'dropout1')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense2')(x)
    x = layers.Dropout(0.2, name = 'dropout2')(x)       
    x = layers.Dense(512, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense3')(x)    
    outputs = layers.Dense(1, 
        activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'VGG19')
    
    if (display_model_information == True):
        model.summary()

    return model

def resnet152v2(input_shape: tuple, weights: str, freeze_convolutional_base: bool,
    display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized ResNet152v2 binary classification model.

    This method `resnet152v2` creates a transfer-learning customized ResNet152v2 binary classification model.

    @param `input_shape`. Dimension of input data in the format of `(height, width, channels)`. Minimum supported dimension is `(32, 32, 3)`.
    @param `weights`. Pretrained model weights.
    @param `freeze_convolutional_base`. Whether to freeze the convolution base.
    @param `display_model_information`. Whether to display model's information.
    """
    
    K.clear_session()
    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    resnet152v2 = \
        ResNet152V2(include_top = False, input_tensor = normalized_augmented, weights = weights)
    if freeze_convolutional_base == True:
        resnet152v2.trainable = False
    else:
        resnet152v2.trainable = True
    x = resnet152v2.output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense1')(x)
    x = layers.Dropout(0.2, name = 'dropout1')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense2')(x)
    x = layers.Dropout(0.2, name = 'dropout2')(x)       
    x = layers.Dense(2048, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense3')(x)    
    outputs = layers.Dense(1, 
        activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'ResNet152v2')

    if (display_model_information == True):
        model.summary()

    return model

def inception_resnetv2(input_shape: tuple, weights: str, freeze_convolutional_base: bool,
    display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized InceptionResNetv2 binary classification model.

    This method inception_resnetv2 creates a transfer-learning customized InceptionResNetv2 binary classification model. If prompted by users, the model's information will be printed on the display.

    @param `input_shape`. Dimension of input data in the format of `(height, width, channels)`. Minimum supported dimension is `(75, 75, 3)`.
    @param `weights`. Pretrained model weights.
    @param `freeze_convolutional_base`. Whether to freeze the convolution base.
    @param `display_model_information`. Whether to display model's information.
    """
    
    K.clear_session()
    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    inception_resnetv2 = InceptionResNetV2(include_top = False, 
        input_tensor = normalized_augmented, weights = weights)
    if freeze_convolutional_base == True:
        inception_resnetv2.trainable = False
    else:
        inception_resnetv2.trainable = True
    x = inception_resnetv2.output
    x = inception_resnetv2.output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense1')(x)
    x = layers.Dropout(0.2, name = 'dropout1')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense2')(x)
    x = layers.Dropout(0.2, name = 'dropout2')(x)       
    x = layers.Dense(2048, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense3')(x)    
    outputs = layers.Dense(1, 
        activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs, name = 'InceptionResNetv2')
    
    if (display_model_information == True):
        model.summary()

    return model

def nasnetlarge(input_shape: tuple, weights: str, freeze_convolutional_base: bool,
    display_model_information: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen

    Create a customized NASNetLarge binary classification model.

    This method efficientnetB7 creates a transfer-learning customized NASNetLarge binary classification model. If prompted by users, the model's information will be printed on the display.

    @param `input_shape`. Dimension of input data in the format of `(height, width, channels)`. Minimum supported dimension is `(32, 32, 3)`.
    @param `weights`. Pretrained model weights.
    @param `freeze_convolutional_base`. Whether to freeze the convolution base.
    @param `display_model_information`. Whether to display model's information.
    """
    
    K.clear_session()

    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    nasnetlarge = NASNetLarge(include_top = False, 
        input_tensor = normalized_augmented, weights = weights)
    if freeze_convolutional_base == True:
        nasnetlarge.trainable = False
    else:
        nasnetlarge.trainable = True
    x = nasnetlarge.output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense1')(x)
    x = layers.Dropout(0.2, name = 'dropout1')(x)
    x = layers.Dense(4096, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense2')(x)
    x = layers.Dropout(0.2, name = 'dropout2')(x)       
    x = layers.Dense(2048, 
        activation = 'relu', kernel_initializer = 'he_normal', name = 'dense3')(x)    
    outputs = layers.Dense(1, 
        activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    if (display_model_information == True):
        model.summary()
    return model

########################
# RECONSTRUCTION MODEL #
########################
def cae_VGG19(input_shape: tuple, weights: str, freeze_convolutional_base: bool,
    display_model_information: bool):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    K.clear_session()

    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    resized = layers.Resizing(height = 512, width = 512, interpolation = 'lanczos5')(normalized_augmented)
    vgg19 = VGG19(include_top = False, input_tensor = resized, weights = weights)

    if freeze_convolutional_base == True:
        vgg19.trainable = False
    else:
        vgg19.trainable = True
    x = vgg19.output
    x = layers.Conv2DTranspose(filters = 512, kernel_size = (3, 3), strides = 2, 
        activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2DTranspose(filters = 512, kernel_size = (3, 3), strides = 2, 
        activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2, 
        activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = 2, 
        activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = 2, 
        activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = layers.Conv2D(filters = input_shape[2], kernel_size = (3, 3), activation = None, 
        padding = 'same', kernel_initializer = 'he_normal')(x)    
    x = layers.Resizing(height = 300, width = 300, interpolation = 'lanczos5')(x)
    outputs = layers.Rescaling(255.0 / 1)(x)

    model = Model(inputs = inputs, outputs = outputs)
     
    if display_model_information == True:
        model.summary()

    return model

def gan_generator():

    return None
def gan_discrimiator(input_shape: tuple, display_model_information: bool):
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    inputs = layers.Input(shape = input_shape)
    normalized_augmented = normalize_and_augmentation(inputs)
    resized = layers.Resizing(height = 512, width = 512, interpolation = 'lanczos5')(normalized_augmented)
    vgg19 = VGG19(include_top = False, input_tensor = resized, weights = 'imagenet')
    vgg19.trainable = True
    x = vgg19.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'outputs')

    model = Model(inputs = inputs, outputs = outputs)

    return model