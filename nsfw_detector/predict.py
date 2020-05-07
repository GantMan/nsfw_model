from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
import numpy as np
import json
from os import listdir
from os.path import isfile, join, exists, isdir, abspath
import tensorflow as tf

IMAGE_DIM = 224
flags.DEFINE_string(
		"image_source", None,
		"A directory of images or a single image to classify.")
flags.DEFINE_string(
		"saved_model_path",
		None,
		"The model to load.")
flags.DEFINE_integer(
    "image_dim",
    IMAGE_DIM,
    "The square dimension of the model's input shape."
    )

FLAGS = flags.FLAGS

def load_images(image_paths, image_size):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]

    for img_path in image_paths:
        try:
            print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)
    
    return np.asarray(loaded_images), loaded_image_paths


def load_model(model_path):
    if model_path is None or not exists(model_path):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    images, image_paths = load_images(input_paths, (image_dim, image_dim))

    model_preds = model.predict(images)
    
    preds = np.argsort(model_preds, axis = 1).tolist()
    
    probs = []

    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    for i, single_preds in enumerate(preds):
        single_probs = []
        for j, pred in enumerate(single_preds):
            single_probs.append(model_preds[i][pred])
            preds[i][j] = categories[pred]
        
        probs.append(single_probs)
    
    images_preds = {}
    
    for i, loaded_image_path in enumerate(image_paths):
        images_preds[loaded_image_path] = {}
        for _ in range(len(preds[i])):
            images_preds[loaded_image_path][preds[i][_]] = str(probs[i][_])
    return images_preds


def main(args):
    del args
    
    if FLAGS.image_source is None or not exists(FLAGS.image_source):
    	raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    model = load_model(FLAGS.saved_model_path)    
    image_preds = classify(model, FLAGS.image_source, FLAGS.image_dim)
    print(json.dumps(image_preds, sort_keys=True, indent=2), '\n')


def run_main():
	app.run(main)

if __name__ == "__main__":
	run_main()