import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data(os.getcwd() + '/data/mnist.npz')

x_train = x_train.astype('float32') / 255. - 0.5
x_test = x_test.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

print x_train.shape
print x_test.shape

encoding_dim = 2

input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, outputs=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True)
# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()
