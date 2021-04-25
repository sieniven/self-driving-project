import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# Setup Keras
from keras.models import Sequential
from keras import layers


# Load pickled data
with open('small_train_traffic.p', mode='rb') as f:
    data = pickle.load(f)

# split data
X_train, y_train= data['features'], data['labels']

#Build Convolutional Pooling Neural Network with Dropout in Keras Here
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='valid'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten(input_shape=(32, 32, 32)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

# compile and train model
# Training for 3 epochs should result in > 50% accuracy
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=25, validation_split=0.2)