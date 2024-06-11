import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from models import VGG16

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
    
      
batch_size = 64  
epochs = 40
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]
model = VGG16()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs)
scores = model.evaluate(x_test, y_test, verbose=1)
