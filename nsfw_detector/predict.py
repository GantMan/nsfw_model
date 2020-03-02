from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from tensorflow import keras
import numpy as np
import os
import json
from os import listdir
from os.path import isfile, join
import tensorflow as tf

flags.DEFINE_string(
		"image_dir", None,
		"A directory of images to classify.")
flags.DEFINE_string(
		"saved_model_path",
		None,
		"The model to load.")
flags.DEFINE_integer(
    "image_dim",
    224,
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

    if os.path.isdir(image_paths):
        print('wut')
        parent = image_paths
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    else:
        print('wut1')
        image_paths = [image_paths]

    for i, img_path in enumerate(image_paths):
        try:
            print(img_path)
            image = keras.preprocessing.image.load_img(img_path, target_size = image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print(i, img_path, ex)
    
    return np.asarray(loaded_images), loaded_image_paths

def main(args):
    del args
    
    if FLAGS.image_dir is None or not os.path.exists(FLAGS.image_dir):
    	raise ValueError("image_dir must be a valid directory with images to classify.")
    
    if FLAGS.saved_model_path is None or not os.path.exists(FLAGS.image_dir):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(FLAGS.saved_model_path)
    
    images, image_paths = load_images(FLAGS.image_dir, (FLAGS.image_dim, FLAGS.image_dim))
    
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
    
    print(json.dumps(images_preds, sort_keys=True, indent=2), '\n')

def run_main():
	app.run(main)

if __name__ == "__main__":
	run_main()