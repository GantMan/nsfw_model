import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import xception
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import initializers, regularizers

# Config
height = 256
width = 256
num_channels = 3
num_classes = 5
weights_file = "weights.best_xception.hdf5"

conv_base = xception.Xception(
      weights='imagenet', 
      include_top=False, 
      input_shape=(height, width, num_channels)
)

# Let's unlock som trainable layers in conv_base
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block14_sepconv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Let's see it
print('Summary')
print(conv_base.summary())

# Let's construct that top layer replacement
x = conv_base.layers[-1].output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.5)(x)
x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.25)(x)
top_layer = Dense(num_classes, activation='softmax')(x)

print('Stacking New Layers')
model=Model(conv_base.layers[0].input, top_layer)

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# checkpoint
filepath = weights_file
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

GENERATOR_BATCH_SIZE = 8
base_dir = 'D:\\nswf_model_training_data\\data'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=GENERATOR_BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=GENERATOR_BATCH_SIZE
)

# Comment in this line if you're looking to reload the last model for training
# Essentially, not taking the best validation weights but to add more epochs
model = load_model('nsfw.h5')

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=100,
    steps_per_epoch=2000,
    shuffle=True,
    # having crazy threading issues
    # set workers to zero if you see an error like: 
    # `freeze_support()`
    workers=0,
    use_multiprocessing=True,
    validation_data=validation_generator,
    validation_steps=200
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
