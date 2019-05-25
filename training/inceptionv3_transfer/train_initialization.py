import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import SGD
from pathlib import Path
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from tensorflow.keras import initializers, regularizers
from tensorflow_model_optimization.sparsity import keras as sparsity

# reusable stuff
import constants
import callbacks
import generators

# No kruft plz
clear_session()
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

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

# Get training/validation data via generators
train_generator, validation_generator = generators.create_generators(height, width)

def model_fn()
    DROPOUT_RATE = Linear('dropout_rate', 0.0, 0.5, 5, group="dense")
    L1_NUM_DIMS = Range('num_dims', 32, 256, 32, group="dense")
    L2_NUM_DIMS = Range('num_dims', 32, 128, 32, group="dense")
    MOMENTUM = Choice('momentum', [0.1, 0.4, 0.8, 0.9, 0.95], group="optimizer")

    # Sparcity 
    pruning_params = {
        'pruning_schedule': sparsity.ConstantSparsity(0.5, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

    # Let's construct that top layer replacement
    x = conv_base.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Flatten()(x)
    x = Dense(L1_NUM_DIMS, activation='relu', kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
    x = Dropout(DROPOUT_RATE)(x)
    # Essential to have another layer for better accuracy
    x = Dense(L2_NUM_DIMS,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
    x = Dropout(DROPOUT_RATE)(x)
    predictions = Dense(constants.NUM_CLASSES,  kernel_initializer="glorot_uniform", activation='softmax')(x)

    print('Stacking New Layers')
    model = sparsity.prune_low_magnitude(Model(inputs = conv_base.input, outputs=predictions), **pruning_params)
    # Normal non pruned model = Model(inputs = conv_base.input, outputs=predictions)

    # Load checkpoint if one is found
    # if os.path.exists(weights_file):
    #         print ("loading ", weights_file)
    #         model.load_weights(weights_file)

    # Get all model callbacks
    callbacks_list = callbacks.make_callbacks(weights_file)

    print('Compile model')
    # originally adam, but research says SGD with scheduler
    # opt = Adam(lr=0.001, amsgrad=True)
    opt = SGD(momentum=MOMENTUM)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    return model

print('Start tuning')
tuner = Tuner(model_fn, 'val_accuracy', epoch_budget=500, max_epochs=10)
tuner.search(tfg.validation_data=validation_generator)

print('Start training!')
history = model.fit_generator(
    train_generator,
    callbacks=callbacks_list,
    epochs=constants.TOTAL_EPOCHS,
    steps_per_epoch=constants.STEPS_PER_EPOCH,
    shuffle=True,
    # having crazy threading issues
    # set workers to zero if you see an error like: 
    # `freeze_support()`
    workers=0,
    use_multiprocessing=True,
    validation_data=validation_generator,
    validation_steps=constants.VALIDATION_STEPS
)

# Save it for later
print('Saving Model - with optimizers')
file_name = "nsfw_optimizers." + str(width) + "x" + str(height) + ".h5"
tf.keras.models.save_model(model, file_name, include_optimizer=True)
# model.save("nsfw_optimizers." + str(width) + "x" + str(height) + ".h5")
