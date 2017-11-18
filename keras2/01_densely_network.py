import numpy as np
import keras

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist

batch_size = 128
num_class = 10
epochs = 12

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype(np.float32)
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

print "x_train shape: ", x_train.shape
print x_train.shape[0], " train smaples"
print x_test.shape[0], " test samples"

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)

print "inputs shape = ", input_shape

inputs = Input(shape=input_shape)
conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu")(conv1)
pool = MaxPool2D(pool_size=(2, 2))(conv2)
dropout1 = Dropout(0.25)(pool)
flatten = Flatten()(dropout1)
dense1 = Dense(128, activation="relu")(flatten)
dropout2 = Dropout(0.5)(dense1)
dense2 = Dense(num_class, activation="softmax")(dropout2)

model = Model(inputs=inputs, outputs=dense2)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)

print "test loss : ", score[0]
print "test accuracy : ", score[1]






