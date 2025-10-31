from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    # For deployment, load labels directly from Labels.txt
    try:
        with open("../Labels.txt") as f:
            labels_list = [line.strip() for line in f.readlines()]
        # Return as dict {0: label0, 1: label1, ...}
        labels = {i: label for i, label in enumerate(labels_list)}
    except Exception:
        # fallback hardcoded labels if Labels.txt missing
        labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
    return labels

def preprocess(image):
    try:
        resample_mode = Image.Resampling.LANCZOS  # Pillow >= 10
    except AttributeError:
        resample_mode = Image.ANTIALIAS           # Pillow < 10

    image = np.array(image.resize((300, 300), resample_mode))
    image = np.array(image, dtype='uint8')
    image = image / 255.0
    return image

def model_arc():
    model = Sequential()

    # Convolution blocks
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(300,300,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Classification layers
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    # Enable OneDNN optimizations
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    return model
