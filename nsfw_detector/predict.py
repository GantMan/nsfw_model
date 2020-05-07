#! python

from absl import app
import argparse
import json
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_DIM = 224   # required/default image dimensionality

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


def main(args=None):
    parser = argparse.ArgumentParser(
        description="""A script to perform NFSW classification of images""",
        epilog="""
        Launch with default model and a test image
            python nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg
    """, formatter_class=argparse.RawTextHelpFormatter)
    
    submain = parser.add_argument_group('main execution and evaluation functionality')
    submain.add_argument('--image_source', dest='image_source', type=str, required=True, 
                            help='A directory of images or a single image to classify')
    submain.add_argument('--saved_model_path', dest='saved_model_path', type=str, required=True, 
                            help='The model to load')
    submain.add_argument('--image_dim', dest='image_dim', type=int, default=IMAGE_DIM,
                            help="The square dimension of the model's input shape")
    if args is not None:
        config = vars(parser.parse_args(args))
    else:
        config = vars(parser.parse_args())

    if config['image_source'] is None or not exists(config['image_source']):
    	raise ValueError("image_source must be a valid directory with images or a single image to classify.")
    
    model = load_model(config['saved_model_path'])    
    image_preds = classify(model, config['image_source'], config['image_dim'])
    print(json.dumps(image_preds, sort_keys=True, indent=2), '\n')


def run_main():
	app.run(main)

if __name__ == "__main__":
	run_main()