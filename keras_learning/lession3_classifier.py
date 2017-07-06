import numpy as np
import os

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


(x_train, y_train), (x_test, y_test) = mnist.load_data(os.getcwd() + '/data/mnist.npz')

x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build neural net
model = Sequential([
    Dense(output_dim=32, input_dim=28*28),
    Activation('relu'),
    Dense(output_dim=10),
    Activation('softmax')
])

rmsprop_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6, decay=0.01)

model.compile(optimizer=rmsprop_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print 'Training ......'
model.fit(x=x_train, y=y_train, batch_size=32, epochs=10)

print 'Testing ......'
loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=32)
print 'loss = %f , accuracy = %f' % (loss, accuracy)