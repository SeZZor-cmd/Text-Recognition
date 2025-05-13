"""import numpy as np
import pandas as pd
import os
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

data_dir = 'D:\MAJOR PROJECT\Dataset'
CATEGORIES = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
              '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',
              '41','42','43','44','45']
def load_unique(DIR):
    images_for_plot = []
    labels_for_plot = []
    size_img = 64,64
    for category in CATEGORIES:
        path = os.path.join(DIR,category)
        class_num = category
        print(category, end = ' | ')
        for img in os.listdir(path):
            image = cv2.imread(os.path.join(path,img))
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            images_for_plot.append(final_img)
            labels_for_plot.append(class_num)
            break
    return images_for_plot, labels_for_plot
images_for_plot, labels_for_plot = load_unique(data_dir)
print("unique_labels = ", labels_for_plot)

fig = plt.figure(figsize = (15,15))
def plot_images(fig, image, label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image, cmap = 'gray')
    plt.title(label)
    return

image_index = 0
row = 5
col = 6
for i in range(1,(row*col)):
    if i > 25:
      break
    plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)
    image_index = image_index + 1
plt.show()

def load_data_train(DIR):
    train_data = []
    size = 32, 32
    print("LOADING DATA FROM : ", end="")
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = CATEGORIES.index(category)
        print(category, end=' | ')
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            try:
                new_array = cv2.resize(img_array, size)
                final_img = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
                train_data.append([final_img, class_num])
            except:
                print(os.path.join(path, img))
    random.shuffle(train_data)
    X = []
    Y = []
    for features, label in train_data:
        X.append(features)
        Y.append(label)
    X = np.array(X).reshape(-1, 32, 32, 1)
    Y = np.array(Y)
    X = X.astype('float32') / 255.0

    Y = to_categorical(Y, 45)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing', 'Test data shape =', X_test.shape)

    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = load_data_train(data_dir)

def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=[3, 3], activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(Conv2D(64, kernel_size=[3, 3], activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(45, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    print("MODEL CREATED")
    model.summary()
    return model

def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    # axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    # axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def fit_model():
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                   save_best_only=True)
    callbacks_list = [checkpointer]
    model_hist = model.fit(X_train, Y_train, batch_size=16, epochs=3, callbacks=callbacks_list,
                           steps_per_epoch=len(X_train) / 16, validation_data=(X_test, Y_test))
    plot_model_history(model_hist)

    return model_hist

model = create_model()

model_hist = fit_model()

model.load_weights('model.weights.best.hdf5')

evaluate_metrics = model.evaluate(X_test, Y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
"""
import numpy as np
import pandas as pd
import os
import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

data_dir = 'D:\MAJOR PROJECT\Dataset'
CATEGORIES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20',
              '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
              '38', '39', '40',
              '41', '42', '43', '44', '45']


def load_unique(DIR):
    images_for_plot = []
    labels_for_plot = []
    size_img = 64, 64
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = category
        print(category, end=' | ')
        for img in os.listdir(path):
            image = cv2.imread(os.path.join(path, img))
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            images_for_plot.append(final_img)
            labels_for_plot.append(class_num)
            break
    return images_for_plot, labels_for_plot


images_for_plot, labels_for_plot = load_unique(data_dir)
print("unique_labels = ", labels_for_plot)

fig = plt.figure(figsize=(15, 15))


def plot_images(fig, image, label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.title(label)
    return


image_index = 0
row = 5
col = 6
for i in range(1, (row * col)):
    if i > 25:
        break
    plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)
    image_index += 1
plt.show()


def load_data_train(DIR):
    train_data = []
    size = 32, 32
    print("LOADING DATA FROM: ", end="")
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        class_num = CATEGORIES.index(category)
        print(category, end=' | ')
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            try:
                new_array = cv2.resize(img_array, size)
                final_img = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)
                train_data.append([final_img, class_num])
            except:
                print(os.path.join(path, img))
    random.shuffle(train_data)
    X = []
    Y = []
    for features, label in train_data:
        X.append(features)
        Y.append(label)
    X = np.array(X).reshape(-1, 32, 32, 1)
    Y = np.array(Y)
    X = X.astype('float32') / 255.0
    Y = to_categorical(Y, 45)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print()
    print('Loaded', len(X_train), 'images for training,', 'Train data shape =', X_train.shape)
    print('Loaded', len(X_test), 'images for testing', 'Test data shape =', X_test.shape)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_data_train(data_dir)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(45, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    print("MODEL CREATED")
    model.summary()
    return model


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


def fit_model():
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    callbacks_list = [checkpointer, early_stopping, reduce_lr]

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    )

    datagen.fit(X_train)

    model_hist = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                           epochs=50,
                           callbacks=callbacks_list,
                           validation_data=(X_test, Y_test))

    plot_model_history(model_hist)
    return model_hist


model = create_model()
model_hist = fit_model()
model.load_weights('model.weights.best.hdf5')

evaluate_metrics = model.evaluate(X_test, Y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1] * 100), "\nEvaluation loss = ",
      "{:.6f}".format(evaluate_metrics[0]))
