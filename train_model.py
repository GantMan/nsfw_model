import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras import initializers, regularizers

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

# Let's unlock som trainable layers
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Let's see it
print('Summary')
print(conv_base.summary())

model = Sequential([
    conv_base,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer=initializers.he_normal(seed=None)),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

if os.path.exists("weights.best.hdf5"):
        print ("loading ", "weights.best.hdf5")
        model.load_weights('weights.best.hdf5')

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('Compile model')
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

GENERATOR_BATCH_SIZE = 32
base_dir = 'D:\\nswf_model_training_data\\data'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    class_mode='categorical',
    batch_size=GENERATOR_BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 512562),
    class_mode='categorical',
    batch_size=GENERATOR_BATCH_SIZE
)

import time
time.sleep(1)

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=50,
    steps_per_epoch=3000,
    shuffle=True,
    # having crazy threading issues
    workers=0,
    use_multiprocessing=True,
    validation_data=validation_generator,
    validation_steps=600
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.figure()

# Save it for later
model.save("nsfw.h5")
