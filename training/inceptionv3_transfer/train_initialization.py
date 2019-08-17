import os
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.applications import InceptionV3
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers

# reusable stuff
import constants
import callbacks
import generators

# No kruft plz
clear_session()

# Config
height = constants.SIZES['basic']
width = height
weights_file = "weights.best_inception" + str(height) + ".hdf5"

conv_base = InceptionV3(
    weights='imagenet', 
    include_top=False, 
    input_shape=(height, width, constants.NUM_CHANNELS)
)

# First time run, no unlocking
conv_base.trainable = False

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
# Essential to have another layer for better accuracy
x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
x = Dropout(0.25)(x)
predictions = Dense(constants.NUM_CLASSES,  kernel_initializer="glorot_uniform", activation='softmax')(x)

print('Stacking New Layers')
model = Model(inputs = conv_base.input, outputs=predictions)

# Load checkpoint if one is found
if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

# Get all model callbacks
callbacks_list = callbacks.make_callbacks(weights_file)

print('Compile model')
# originally adam, but research says SGD with scheduler
# opt = Adam(lr=0.001, amsgrad=True)
opt = SGD(momentum=.9)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    workers=4,
    use_multiprocessing=False,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS
)

# Save it for later
print('Saving Model')
model.save("nsfw." + str(width) + "x" + str(height) + ".h5")
