# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a TensorFlow model based on directories of images.

This program builds, trains and exports a TensorFlow 2.x model that classifies
natural images (photos) into a fixed set of classes. The classes are learned
from a user-supplied dataset of images, stored as a directory of subdirectories
of JPEG images, each subdirectory representing one class.

The model is built from a pre-trained image feature vector module from
TensorFlow Hub (in its TF2/SavedModel format, not the older hub.Module format)
followed by a linear classifier. The linear classifier, and optionally also
the TF Hub module, are trained on the new dataset. TF Hub offers a variety of
suitable modules with various size/accuracy tradeoffs.

The resulting model can be exported in TensorFlow's standard SavedModel format
and as a .tflite file for deployment to mobile devices with TensorFlow Lite.
TODO(b/139467904): Add support for post-training model optimization.

For more information, please see the README file next to the source code,
https://github.com/tensorflow/hub/blob/master/tensorflow_hub/tools/make_image_classifier/README.md
"""

# NOTE: This is an expanded, command-line version of
# https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb
# PLEASE KEEP THEM IN SYNC, such that running tests for this program
# provides assurance that the code in the colab notebook works.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
import collections
import copy
import make_nsfw_model_lib as lib
import numpy as np
import os
import re
import six
import tempfile
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub

_DEFAULT_HPARAMS = lib.get_default_hparams()

flags.DEFINE_string(
		"image_dir", None,
		"A directory with subdirectories of images, one per class. "
		"If unset, the TensorFlow Flowers example dataset will be used. "
		"Internally, the dataset is split into training and validation pieces.")
flags.DEFINE_string(
		"tfhub_module",
		"https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
		"Which TF Hub module to use. Must be a module in TF2/SavedModel format "
		"for computing image feature vectors.")
flags.DEFINE_integer(
		"image_size", None,
		"The height and width of images to feed into --tfhub_module. "
		"(For now, must be set manually for modules with variable input size.)")
flags.DEFINE_string(
		"saved_model_dir", None,
		"The final model is exported as a SavedModel directory with this name.")
flags.DEFINE_string(
		"tflite_output_file", None,
		"The final model is exported as a .tflite flatbuffers file with this name.")
flags.DEFINE_string(
		"labels_output_file", None,
		"Where to save the labels (that is, names of image subdirectories). "
		"The lines in this file appear in the same order as the predictions "
		"of the model.")
flags.DEFINE_float(
		"assert_accuracy_at_least", None,
		"If set, the program fails if the validation accuracy at the end of "
		"training is less than this number (between 0 and 1), and no export of "
		"the trained model happens.")
flags.DEFINE_integer(
		"train_epochs", _DEFAULT_HPARAMS.train_epochs,
		"Training will do this many iterations over the dataset.")
flags.DEFINE_bool(
		"do_fine_tuning", _DEFAULT_HPARAMS.do_fine_tuning,
		"If set, the --tfhub_module is trained together with the rest of "
		"the model being built.")
flags.DEFINE_integer(
		"batch_size", _DEFAULT_HPARAMS.batch_size,
		"Each training step samples a batch of this many images "
		"from the training data. (You may need to shrink this when using a GPU "
		"and getting out-of-memory errors. Avoid values below 8 when re-training "
		"modules that use batch normalization.)")
flags.DEFINE_float(
		"learning_rate", _DEFAULT_HPARAMS.learning_rate,
		"The learning rate to use for gradient descent training.")
flags.DEFINE_float(
		"momentum", _DEFAULT_HPARAMS.momentum,
		"The momentum parameter to use for gradient descent training.")
flags.DEFINE_float(
		"dropout_rate", _DEFAULT_HPARAMS.dropout_rate,
		"The fraction of the input units to drop, used in dropout layer.")
flags.DEFINE_bool(
		"is_deprecated_tfhub_module", False,
		"Whether or not the supplied TF hub module is old and from Tensorflow 1.")
flags.DEFINE_float(
		"label_smoothing", _DEFAULT_HPARAMS.label_smoothing,
		"The degree of label smoothing to use.")
flags.DEFINE_float(
		"validation_split", _DEFAULT_HPARAMS.validation_split,
		"The percentage of data to use for validation.")
flags.DEFINE_string(
    'optimizer', _DEFAULT_HPARAMS.optimizer,
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
flags.DEFINE_float(
    'adadelta_rho', _DEFAULT_HPARAMS.adadelta_rho,
    'The decay rate for adadelta.')
flags.DEFINE_float(
    'adagrad_initial_accumulator_value', _DEFAULT_HPARAMS.adagrad_initial_accumulator_value,
    'Starting value for the AdaGrad accumulators.')
flags.DEFINE_float(
    'adam_beta1', _DEFAULT_HPARAMS.adam_beta1,
    'The exponential decay rate for the 1st moment estimates.')
flags.DEFINE_float(
    'adam_beta2', _DEFAULT_HPARAMS.adam_beta2,
    'The exponential decay rate for the 2nd moment estimates.')
flags.DEFINE_float('opt_epsilon', _DEFAULT_HPARAMS.opt_epsilon, 'Epsilon term for the optimizer.')
flags.DEFINE_float('ftrl_learning_rate_power', _DEFAULT_HPARAMS.ftrl_learning_rate_power,
                          'The learning rate power.')
flags.DEFINE_float(
    'ftrl_initial_accumulator_value', _DEFAULT_HPARAMS.ftrl_initial_accumulator_value,
    'Starting value for the FTRL accumulators.')
flags.DEFINE_float(
    'ftrl_l1', _DEFAULT_HPARAMS.ftrl_l1, 'The FTRL l1 regularization strength.')

flags.DEFINE_float(
    'ftrl_l2', _DEFAULT_HPARAMS.ftrl_l2, 'The FTRL l2 regularization strength.')
flags.DEFINE_float('rmsprop_momentum', _DEFAULT_HPARAMS.rmsprop_momentum, 'Momentum.')
flags.DEFINE_float('rmsprop_decay', _DEFAULT_HPARAMS.rmsprop_decay, 'Decay term for RMSProp.')
flags.DEFINE_bool(
		"do_data_augmentation", False,
		"Whether or not to do data augmentation.")
flags.DEFINE_bool(
		"use_mixed_precision", False,
		"Whether or not to use NVIDIA mixed precision. Requires NVIDIA card with at least compute level 7.0")

FLAGS = flags.FLAGS


def _get_hparams_from_flags():
	"""Creates dict of hyperparameters from flags."""
	return lib.HParams(
			train_epochs=FLAGS.train_epochs,
			do_fine_tuning=FLAGS.do_fine_tuning,
			batch_size=FLAGS.batch_size,
			learning_rate=FLAGS.learning_rate,
			momentum=FLAGS.momentum,
			dropout_rate=FLAGS.dropout_rate,
			label_smoothing=FLAGS.label_smoothing,
			validation_split=FLAGS.validation_split,
			optimizer=FLAGS.optimizer,
			adadelta_rho=FLAGS.adadelta_rho,
			adagrad_initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value,
			adam_beta1=FLAGS.adam_beta1,
			adam_beta2=FLAGS.adam_beta2,
			opt_epsilon=FLAGS.opt_epsilon,
			ftrl_learning_rate_power=FLAGS.ftrl_learning_rate_power,
			ftrl_initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
			ftrl_l1=FLAGS.ftrl_l1,
			ftrl_l2=FLAGS.ftrl_l2,
			rmsprop_momentum=FLAGS.rmsprop_momentum,
			rmsprop_decay=FLAGS.rmsprop_decay,
			do_data_augmentation=FLAGS.do_data_augmentation,
			use_mixed_precision=FLAGS.use_mixed_precision
			)
			



def _check_keras_dependencies():
	"""Checks dependencies of tf.keras.preprocessing.image are present.

	This function may come to depend on flag values that determine the kind
	of preprocessing being done.

	Raises:
		ImportError: If dependencies are missing.
	"""
	try:
		tf.keras.preprocessing.image.load_img(six.BytesIO())
	except ImportError:
		print("\n*** Unsatisfied dependencies of keras_preprocessing.image. ***\n"
					"To install them, use your system's equivalent of\n"
					"pip install tensorflow_hub[make_image_classifier]\n")
		raise
	except Exception as e:	# pylint: disable=broad-except
		# Loading from dummy content as above is expected to fail in other ways.
		pass


def _assert_accuracy(train_result, assert_accuracy_at_least):
	# Fun fact: With TF1 behavior, the key was called "val_acc".
	val_accuracy = train_result.history["val_accuracy"][-1]
	accuracy_message = "found {:f}, expected at least {:f}".format(
			val_accuracy, assert_accuracy_at_least)
	if val_accuracy >= assert_accuracy_at_least:
		print("ACCURACY PASSED:", accuracy_message)
	else:
		raise AssertionError("ACCURACY FAILED:", accuracy_message)

def main(args):
	"""Main function to be called by absl.app.run() after flag parsing."""
	del args

	#policy = mixed_precision.Policy('mixed_float16')
	#mixed_precision.set_policy(policy)

#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(False)
	physical_devices = tf.config.list_physical_devices('GPU') 
	try: 
		tf.config.experimental.set_memory_growth(physical_devices[0], True) 
		print('Configured device')
	except: 
		# Invalid device or cannot modify virtual devices once initialized. 
		pass

	_check_keras_dependencies()
	hparams = _get_hparams_from_flags()

	image_dir = FLAGS.image_dir or lib.get_default_image_dir()

	model, labels, train_result, frozen_graph = lib.make_image_classifier(
			FLAGS.tfhub_module, image_dir, hparams, FLAGS.image_size, FLAGS.saved_model_dir)
	if FLAGS.assert_accuracy_at_least:
		_assert_accuracy(train_result, FLAGS.assert_accuracy_at_least)
	print("Done with training.")

	if FLAGS.labels_output_file:
		labels_dir_path = os.path.dirname(FLAGS.labels_output_file)
		# Ensure dir structure exists
		Path(labels_dir_path).mkdir(parents=True, exist_ok=True)
		with tf.io.gfile.GFile(FLAGS.labels_output_file, "w") as f:
			f.write("\n".join(labels + ("",)))
		print("Labels written to", FLAGS.labels_output_file)

	saved_model_dir = FLAGS.saved_model_dir

	if FLAGS.tflite_output_file and not saved_model_dir:
		# We need a SavedModel for conversion, even if the user did not request it.
		saved_model_dir = tempfile.mkdtemp()

	if saved_model_dir:
		# Ensure dir structure exists
		Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
		tf.saved_model.save(model, saved_model_dir)
		keras_model_path = os.path.join(saved_model_dir, "saved_model.h5")
		weights_path = os.path.join(saved_model_dir, "saved_model_weights.h5")
		model.save(keras_model_path)
		model.save_weights(weights_path)
		print("SavedModel model exported to", saved_model_dir)

	if FLAGS.tflite_output_file:
		tflite_dir_path = os.path.dirname(FLAGS.tflite_output_file)
		# Ensure dir structure exists
		Path(tflite_dir_path).mkdir(parents=True, exist_ok=True)
		converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
		lite_model_content = converter.convert()
		with tf.io.gfile.GFile(FLAGS.tflite_output_file, "wb") as f:
			f.write(lite_model_content)
		print("TFLite model exported to", FLAGS.tflite_output_file)

	if saved_model_dir:
		# Save the frozen graph
		# Ensure dir structure exists
		Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
		tf.io.write_graph(graph_or_graph_def=frozen_graph,
		logdir=saved_model_dir,
		name="frozen_graph.pb",
		as_text=False)


def _ensure_tf2():
	"""Ensure running with TensorFlow 2 behavior.

	This function is safe to call even before flags have been parsed.

	Raises:
		ImportError: If tensorflow is too old for proper TF2 behavior.
	"""
	logging.info("Running with tensorflow %s (git version %s) and hub %s",
							 tf.__version__, tf.__git_version__, hub.__version__)
	if tf.__version__.startswith("1."):
		if tf.__git_version__ == "unknown":	# For internal testing use.
			try:
				tf.compat.v1.enable_v2_behavior()
				return
			except AttributeError:
				pass	# Fail below for missing enabler function.
		raise ImportError("Sorry, this program needs TensorFlow 2.")


def run_main():
	"""Entry point equivalent to executing this file."""
	_ensure_tf2()
	app.run(main)


if __name__ == "__main__":
	run_main()
