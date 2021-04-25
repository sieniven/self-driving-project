x_train = np.array(images)
y_train = np.array(measurements)

from keras.model import Sequential
from keras.layers import Flatten, dense, Lambda

model = Sequential()
# lambda layer for normalization
model.add(Lambda(Lambda x: x / 255.0, input_shape = (160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)