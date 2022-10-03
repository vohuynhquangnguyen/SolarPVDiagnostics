1. Implement classification models (check source code at `model_development.py`) with PyTorch framework as follows:
    * VGG19
    * ResNet152v2
    * InceptionResNetv2
    * EfficientNetB7

2. Train the models with the defined train, validation, and test sets (check source code `VGG19.ipynb` for how to prepare the data)
    * Each model is trained with 2 phases: first, freeze the convolutional base, only train the fully connected layers. Next, unfreeze all layers, train the entire model.
    * First phase use Nadam. Second phase use Adam(learning_rate = 1e-5)