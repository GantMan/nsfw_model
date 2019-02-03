import os
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import InceptionV3
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import initializers, regularizers

# No kruft plz
clear_session()

# Config
height = 299
width = height
num_channels = 3
num_classes = 5
GENERATOR_BATCH_SIZE = 16
weights_file = "weights.best_inception" + str(height) + ".hdf5"

# conv_base = xception.Xception(
#       weights='imagenet', 
#       include_top=False, 
#       input_shape=(height, width, num_channels)
# )

conv_base = InceptionV3(
    weights='imagenet', 
    include_top=False, 
    input_shape=(height, width, num_channels)
)


# base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
# x = base_model.output
# x = AveragePooling2D(pool_size=(8, 8))(x)
# x = Dropout(.4)(x)
# x = Flatten()(x)
# predictions = Dense(n_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

# model = Model(input=base_model.input, output=predictions)


# First time run, no unlocking
conv_base.trainable = False
# Let's unlock trainable layers in conv_base
# conv_base.trainable = True

# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block14_sepconv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# Let's see it
print('Summary')
print(conv_base.summary())

# Let's construct that top layer replacement
x = conv_base.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x - Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
x = Dropout(0.5)(x)
# x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
# x = Dropout(0.25)(x)
predictions = Dense(num_classes,  kernel_initializer="glorot_uniform", activation='softmax')(x)

print('Stacking New Layers')
model=Model(inputs = conv_base.input, outputs=predictions)

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# checkpoint
filepath = weights_file
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Update info
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

callbacks_list = [checkpoint, tensorboard]

print('Compile model')
adam = Adam(lr=0.001, amsgrad=True)
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=30,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

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
# model = load_model("nsfw." + str(width) + "x" + str(height) + ".h5")

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=50,
    steps_per_epoch=100,
    shuffle=True,
    # having crazy threading issues
    # set workers to zero if you see an error like: 
    # `freeze_support()`
    workers=0,
    use_multiprocessing=True,
    validation_data=validation_generator,
    validation_steps=50
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
model.save("nsfw." + str(width) + "x" + str(height) + ".h5")
