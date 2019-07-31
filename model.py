from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.image as mpimg
import cv2


def train(train_dir=r'./small/train', validation_dir=r'./small/validation'):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    # 调整像素值
    train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest'
                                       )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(

        directory=train_dir,
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=6,
        validation_data=validation_generator,
        validation_steps=1)

    model.save('IDcard.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    lena = mpimg.imread('./data/train/0.png')
    lena = cv2.resize(lena, (150, 150))
    print(model.predict(np.asarray([lena])))
    lena = mpimg.imread('./data/train/1.png')
    lena = cv2.resize(lena, (150, 150))
    print(model.predict(np.asarray([lena])))
    lena = mpimg.imread('./data/train/2.png')
    lena = cv2.resize(lena, (150, 150))
    print(model.predict(np.asarray([lena])))


if __name__ == '__main__':
    train()
