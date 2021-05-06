import os
import sys
import keras
import tarfile
import numpy as np
import pandas as pd
import tensorflow as tf
import urllib.request as urllib
import matplotlib.pyplot as plt
import glob

from PIL import Image
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.engine.training import Model
from keras import backend as K, regularizers
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Add, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from sklearn.model_selection import train_test_split

initial_lr = 0.01

dir = '../data/dataset2'
categories = sorted(os.listdir(dir))
nb_classes = len(categories)

X = []
Y = []

image_w = 64
image_h = 64

for idx, f in enumerate(categories):
    label = [0 for _ in range(nb_classes)]
    label[idx] = 1
    image_dir = dir + "/" + f
    files = glob.glob(image_dir + "/*.jpg")

    for i, fname in enumerate(files):
        img = Image.open(fname)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

        # for ang in range(-10, 10, 5):
        #     img2 = img.rotate(ang)
        #     data = np.asarray(img2)
        #     X.append(data)
        #     Y.append(label)
        #
        #     img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        #     data = np.asarray(img2)
        #     X.append(data)
        #     Y.append(label)

X = np.array(X) / 255.0
Y = np.array(Y)

# df = pd.DataFrame({'image': [X], 'class': [Y]})
# df = df.sample(frac=1)
# print(df.head())
# df.to_pickle('dataset.pickle')
x_train, x_test, y_train, y_test = train_test_split(X, Y)

print('X_train shape: ', x_train.shape[0])
print('Y_train shape: ', y_train.shape)

# applying transformation to image
train_gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.5,
    height_shift_range=0.5,
    brightness_range=[0.25, 1.5],
    horizontal_flip=True,)

# test_gen = ImageDataGenerator()
train_gen.fit(x_train)
test_set = train_gen.flow(x_test, y_test, batch_size=256)


model = Sequential()
# Block 1
model.add(Conv2D(32, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same', input_shape=(64, 64, 3)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Block 2
model.add(Conv2D(64, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(2, 2))

# Block 3
model.add(Conv2D(128, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Block 4
model.add(Conv2D(256, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(2, 2))

# Block 5
model.add(Conv2D(512, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

# Block 6
model.add(Conv2D(1024, kernel_size=3, kernel_initializer='he_uniform', kernel_regularizer=l2(0.0005), padding='same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(2, 2))

model.add(Flatten())

# Dense 1
model.add(Dense(1400, kernel_regularizer=l2(0.0005)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Dense 2
model.add(Dense(1, kernel_regularizer=l2(0.0005), activation='softmax'))

# Visualize Model
model.summary()


def lr_scheduler(epoch):
    if epoch < 20:
        return initial_lr
    elif epoch < 40:
        return initial_lr / 20
    elif epoch < 50:
        return initial_lr / 40
    elif epoch < 60:
        return initial_lr / 80
    elif epoch < 70:
        return initial_lr / 160
    elif epoch < 80:
        return initial_lr / 320
    elif epoch < 90:
        return initial_lr / 640
    else:
        return initial_lr / 1280

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(initial_lr),
    metrics = ['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=300,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler)],
    shuffle=True
)

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]*100}')
# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()
# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()