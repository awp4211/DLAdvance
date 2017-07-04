import argparse

def softmax_demo():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    # generate dummy data
    import numpy as np
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)

    print score


def mlp_demo():
    """
    MLP Binary classify
    :return:
    """
    import keras
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    # generate dummy data
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))

    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)


def vgg_demo():
    import numpy as np
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.optimizers import SGD

    # generate dummy data
    x_train = np.random.random((100, 100, 100, 3))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

    model = Sequential()
    # input: 100*100 image with 3 channels -> (100, 100, 3) tensors
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)


def lstm_demo():
    """
    LSTM in keras
    :return:
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='select a demo to run', default='mlp', type=str)
    args = parser.parse_args()
    print args

    if args.model == 'softmax':
        softmax_demo()
    elif args.model == 'mlp':
        mlp_demo()
    elif args.model == 'vgg':
        vgg_demo()


