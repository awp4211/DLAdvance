import numpy as np
import os

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Convolution2D
from keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = mnist.load_data(os.getcwd() + '/data/mnist.npz')

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))

model.add(Dense(10))
model.add(Activation('softmax'))


# Another way to define optimizer

adam_optimizer = Adam(lr=1e-4)

model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print 'Training ......'
model.fit(x_train, y_train, epochs=10, batch_size=32)

print 'Testing .......'
loss, accuracy = model.evaluate(x_test, y_test)

print 'test loss : ', loss
print 'test accuracy : ', accuracy
