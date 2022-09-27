import tensorflow as tf
from keras import Model, layers
from keras.utils import plot_model
from keras.applications import VGG19, ResNet152V2, InceptionResNetV2, EfficientNetB7
import keras.backend as K

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