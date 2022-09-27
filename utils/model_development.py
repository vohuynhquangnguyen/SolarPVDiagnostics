import time
import tensorflow as tf
import numpy as np
from keras import Model, layers
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.applications import VGG19, ResNet152V2, InceptionResNetV2, EfficientNetB7
import keras.backend as K

###########
# METHODS #
###########

def configure_settings():
    """
    @author: Vo, Huynh Quang Nguyen
    """    
    
    physical_devices  =  tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as error:
            print(error)

    tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32 = True)
    
    return None


def train_model(model: object, X: object, Y: object, 
    metric_to_monitor: str, target_metrics: list, loss: object,
    optimizer: object, no_of_epochs: int, batch_size: int,
    date: str, model_name: str, image_shape: tuple):
    """
    @author: Vo, Huynh Quang Nguyen
    """
	
    start_time = time.time()
    ###
    filepath = f'weights/{date}_{model_name}_{image_shape[0]}x{image_shape[1]}x{image_shape[2]}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor = metric_to_monitor, verbose = 1, 
        save_best_only = True, mode = 'max')
    
    
    callbacks_list = [checkpoint]
    model.compile(optimizer = optimizer, loss = loss, metrics = target_metrics)
    history = model.fit(X, Y, validation_split = 0.20, epochs = no_of_epochs, batch_size = batch_size, callbacks = callbacks_list, verbose = 1)
    np.save(f'{model_name}_history.npy', history.history)
    ###
    end_time = time.time()
    training_time = round(end_time - start_time, 4)

    return history, training_time

########################
# CLASSIFICATION MODEL #
########################
def vgg19(input_shape: tuple, model_name: str, visualize_model: bool = False) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()
		
    inputs = layers.Input(shape = input_shape)
    convolutional_base = VGG19(include_top = False, input_tensor = inputs, weights = 'imagenet')
    convolutional_base.trainable = True
    x = convolutional_base.get_layer('block5_conv4').output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense3')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs, name = model_name)

    if (visualize_model == True):
        plot_model(model, to_file = f'./docs/{model_name}.png', show_shapes = True, show_dtype = True, 
        show_layer_names = True)

    return model 

def resnet152v2(input_shape: tuple, model_name: str, visualize_model: bool) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    
    K.clear_session()
		
    inputs = layers.Input(shape = input_shape)
    convolutional_base = ResNet152V2(include_top = False, input_tensor = inputs, weights = 'imagenet')
    convolutional_base.trainable = True
    x = convolutional_base.get_layer(-1).output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense3')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    if (visualize_model == True):
        plot_model(model, to_file = f'./docs/{model_name}.png', show_shapes = True, show_dtype = True, 
        show_layer_names = True)

    return model

def inception_resnetv2(input_shape: tuple, model_name: str, visualize_model: bool = False) -> object:
    """
    @author: Vo, Huynh Quang Nguyen
    """
    K.clear_session()

    inputs = layers.Input(shape = input_shape)
    convolutional_base = InceptionResNetV2(include_top = False, input_tensor = inputs, weights = 'imagenet')
    convolutional_base.trainable = True
    x = convolutional_base.get_layer(-1).output
    x = layers.GlobalAveragePooling2D(name = 'globavgpool')(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense3')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    if (visualize_model == True):
        plot_model(model, to_file = f'./docs/{model_name}.png', show_shapes = True, show_dtype = True, 
        show_layer_names = True)

    return model

def efficientnetB7(input_shape: tuple, model_name: str, visualize_model: bool = False) -> object:
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
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(4096, activation = 'relu', name = 'dense2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation = 'relu', name = 'dense3')(x)    
    outputs = layers.Dense(1, activation = 'sigmoid', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    
    if (visualize_model == True):
        plot_model(model, to_file = f'./docs/{model_name}.png', show_shapes = True, show_dtype = True, 
        show_layer_names = True)
    return model

########################
# RECONSTRUCTION MODEL #
########################