from keras.datasets import cifar10
from keras.utils import np_utils

nb_classes = 10

def load_dataset():
   # the data, shuffled and split between train and test sets
   (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   print('X_train shape:', X_train.shape)
   print(X_train.shape[0], 'train samples')
   print(X_test.shape[0], 'test samples')

   # convert class vectors to binary class matrices
   Y_train = np_utils.to_categorical(y_train, nb_classes)
   Y_test = np_utils.to_categorical(y_test, nb_classes)

   X_train = X_train.astype('float32')
   X_test = X_test.astype('float32')
   X_train /= 255
   X_test /= 255

   return X_train, Y_train, X_test, Y_test

   from keras.models import Sequential
   from keras.layers.core import Dense, Dropout, Activation, Flatten
   from keras.layers.convolutional import Convolution2D, MaxPooling2D

def make_network():
  model = Sequential()

  model.add(Convolution2D(32, 3, 3, border_mode='same',
                          input_shape=(img_channels, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(Convolution2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  return model

def train_model(model, X_train, Y_train, X_test, Y_test):

   sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   model.compile(loss='categorical_crossentropy', optimizer=sgd)

   model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,
             validation_split=0.1, show_accuracy=True, verbose=1)

   print('Testing...')
   res = model.evaluate(X_test, Y_test,
                        batch_size=batch_size, verbose=1, show_accuracy=True)
   print('Test accuracy: {0}'.format(res[1]))
