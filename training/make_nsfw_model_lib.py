# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains a TensorFlow model based on directories of images.

This library provides the major pieces for make_image_classifier (see there).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import multiprocessing
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.keras import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
import collections
import copy
import numpy as np
import os
import re
import six
import tempfile
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub


_DEFAULT_IMAGE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# From https://github.com/tensorflow/hub/issues/390#issuecomment-544489095
# Woops, this doesn't actually work. Sad face emoji.
class Wrapper(tf.train.Checkpoint):
	def __init__(self, spec):
		super(Wrapper, self).__init__()
		self.module = hub.load(spec, tags=[])
		self.variables = self.module.variables
		self.trainable_variables = []
	def __call__(self, x):
		return self.module.signatures["default"](x)["default"]

def get_default_image_dir():
	"""Returns the path to a default image dataset, downloading it if needed."""
	return tf.keras.utils.get_file("flower_photos",
								 _DEFAULT_IMAGE_URL, untar=True)

def configure_optimizer(hparams):
	"""Configures the optimizer used for training.
	
	Args:
		learning_rate: A scalar or `Tensor` learning rate.
	
	Returns:
		An instance of an optimizer.
	
	Raises:
		ValueError: if hparams.optimizer is not recognized.
	"""
	if hparams.optimizer == 'adadelta':
		optimizer = tf.keras.optimizers.Adadelta(
			hparams.learning_rate,
			rho=hparams.adadelta_rho,
			epsilon=hparams.opt_epsilon)
	elif hparams.optimizer == 'adagrad':
		optimizer = tf.keras.optimizers.Adagrad(
			hparams.learning_rate,
			initial_accumulator_value=hparams.adagrad_initial_accumulator_value)
	elif hparams.optimizer == 'adam':
		optimizer = tf.keras.optimizers.Adam(
			hparams.learning_rate,
			beta_1=hparams.adam_beta1,
			beta_2=hparams.adam_beta2,
			epsilon=hparams.opt_epsilon)
	elif hparams.optimizer == 'ftrl':
		optimizer = tf.keras.optimizers.Ftrl(
			hparams.learning_rate,
			learning_rate_power=hparams.ftrl_learning_rate_power,
			initial_accumulator_value=hparams.ftrl_initial_accumulator_value,
			l1_regularization_strength=hparams.ftrl_l1,
			l2_regularization_strength=hparams.ftrl_l2)  
	elif hparams.optimizer == 'rmsprop':
		optimizer = tf.keras.optimizers.RMSprop(learning_rate=hparams.learning_rate, epsilon=hparams.opt_epsilon, momentum=hparams.rmsprop_momentum)	
	elif hparams.optimizer == 'sgd':
		optimizer = tf.keras.optimizers.SGD(learning_rate=hparams.learning_rate, momentum=hparams.momentum)
	else:
		raise ValueError('Optimizer [%s] was not recognized' % hparams.optimizer)
	return optimizer


class HParams(
	collections.namedtuple("HParams", [
		"train_epochs", "do_fine_tuning", "batch_size", "learning_rate",
		"momentum", "dropout_rate", "label_smoothing", "validation_split",
		"optimizer", "adadelta_rho", "adagrad_initial_accumulator_value",
		"adam_beta1", "adam_beta2", "opt_epsilon", "ftrl_learning_rate_power",
		"ftrl_initial_accumulator_value", "ftrl_l1", "ftrl_l2", "rmsprop_momentum",
		"rmsprop_decay", "do_data_augmentation", "use_mixed_precision"
	])):
	"""The hyperparameters for make_image_classifier.

	train_epochs: Training will do this many iterations over the dataset.
	do_fine_tuning: If true, the Hub module is trained together with the
	classification layer on top.
	batch_size: Each training step samples a batch of this many images.
	learning_rate: The learning rate to use for gradient descent training.
	momentum: The momentum parameter to use for gradient descent training.
	dropout_rate: The fraction of the input units to drop, used in dropout layer.
"""


def get_default_hparams():
	"""Returns a fresh HParams object initialized to default values."""
	return HParams(
		train_epochs=5,
		do_fine_tuning=False,
		batch_size=32,
		learning_rate=0.005,
		momentum=0.9,
		dropout_rate=0.2,
		label_smoothing=0.1,
		validation_split=.20,
		optimizer='rmsprop',
		adadelta_rho=0.95,
		adagrad_initial_accumulator_value=0.1,
		adam_beta1=0.9,
		adam_beta2=0.999,
		opt_epsilon=1.0,
		ftrl_learning_rate_power=-0.5,
		ftrl_initial_accumulator_value=0.1,
		ftrl_l1=0.0,
		ftrl_l2=0.0,
		rmsprop_momentum=0.9,
		rmsprop_decay=0.9,
		do_data_augmentation=False,
		use_mixed_precision=False
		)


def _get_data_with_keras(image_dir, image_size, batch_size,
						 validation_size=0.2, do_data_augmentation=False):
	"""Gets training and validation data via keras_preprocessing.

	Args:
	image_dir: A Python string with the name of a directory that contains
		subdirectories of images, one per class.
	image_size: A list or tuple with 2 Python integers specifying
		the fixed height and width to which input images are resized.
	batch_size: A Python integer with the number of images per batch of
		training and validation data.
	do_data_augmentation: An optional boolean, controlling whether the
		training dataset is augmented by randomly distorting input images.

	Returns:
	A nested tuple ((train_data, train_size),
					(valid_data, valid_size), labels) where:
	train_data, valid_data: Generators for use with Model.fit_generator,
		each yielding tuples (images, labels) where
		images is a float32 Tensor of shape [batch_size, height, width, 3]
			with pixel values in range [0,1],
		labels is a float32 Tensor of shape [batch_size, num_classes]
			with one-hot encoded classes.
	train_size, valid_size: Python integers with the numbers of training
		and validation examples, respectively.
	labels: A tuple of strings with the class labels (subdirectory names).
		The index of a label in this tuple is the numeric class id.
	"""
	datagen_kwargs = dict(rescale=1./255,
						# TODO(b/139467904): Expose this as a flag.
						validation_split=validation_size)
	dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size,
						 interpolation="bilinear")

	valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		**datagen_kwargs)
	valid_generator = valid_datagen.flow_from_directory(
		image_dir, subset="validation", shuffle=False, **dataflow_kwargs)

	if do_data_augmentation:
		# TODO(b/139467904): Expose the following constants as flags.
		train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			rotation_range=40, horizontal_flip=True, width_shift_range=0.2,
			height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
			**datagen_kwargs)
	else:
		train_datagen = valid_datagen

	train_generator = train_datagen.flow_from_directory(
			image_dir, subset="training", shuffle=True, **dataflow_kwargs)

	indexed_labels = [(index, label)
					for label, index in train_generator.class_indices.items()]
	sorted_indices, sorted_labels = zip(*sorted(indexed_labels))
	assert sorted_indices == tuple(range(len(sorted_labels)))
	return ((train_generator, train_generator.samples),
			(valid_generator, valid_generator.samples),
			sorted_labels)


def _image_size_for_module(module_layer, requested_image_size=None):
	"""Returns the input image size to use with the given module.

	Args:
	module_layer: A hub.KerasLayer initialized from a Hub module expecting
		image input.
	requested_image_size: An optional Python integer with the user-requested
		height and width of the input image; or None.

	Returns:
	A tuple (height, width) of Python integers that can be used as input
	image size for the given module_layer.

	Raises:
	ValueError: If requested_image_size is set but incompatible with the module.
	ValueError: If the module does not specify a particular inpurt size and
		 requested_image_size is not set.
	"""
	# TODO(b/139530454): Use a library helper function once available.
	# The stop-gap code below assumes any concrete function backing the
	# module call will accept a batch of images with the one accepted size.

	module_image_size = tuple(
	module_layer._func.__call__	# pylint:disable=protected-access
	.concrete_functions[0].structured_input_signature[0][0].shape[1:3])
	
	if requested_image_size is None:
		if None in module_image_size:
			raise ValueError("Must specify an image size because "
							 "the selected TF Hub module specifies none.")
		else:
			return module_image_size
	else:
		requested_image_size = tf.TensorShape([requested_image_size, requested_image_size])
		assert requested_image_size.is_fully_defined()

	if requested_image_size.is_compatible_with(module_image_size):
		return tuple(requested_image_size.as_list())
	else:
		raise ValueError("The selected TF Hub module expects image size {}, "
						 "but size {} is requested".format(
							 module_image_size,
							 tuple(requested_image_size.as_list())))


def build_model(module_layer, hparams, image_size, num_classes):
	"""Builds the full classifier model from the given module_layer.

	Args:
	module_layer: Pre-trained tfhub model layer.
	hparams: A namedtuple of hyperparameters. This function expects
		.dropout_rate: The fraction of the input units to drop, used in dropout
		layer.
	image_size: The input image size to use with the given module layer.
	num_classes: Number of the classes to be predicted.

	Returns:
	The full classifier model.
	"""
	# TODO(b/139467904): Expose the hyperparameters below as flags.

	if hparams.dropout_rate is not None and hparams.dropout_rate > 0:
		model = tf.keras.Sequential([
				tf.keras.Input(shape=(image_size[0], image_size[1], 3), name='input', dtype='float32'), 
				module_layer,
				tf.keras.layers.Dropout(rate=hparams.dropout_rate),
				tf.keras.layers.Dense(
					num_classes,
					kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
				tf.keras.layers.Activation('softmax', dtype='float32', name='prediction')
			])
	else:
		model = tf.keras.Sequential([
				tf.keras.Input(shape=(image_size[0], image_size[1], 3), name='input', dtype='float32'), 
				module_layer,
				tf.keras.layers.Dense(
					num_classes,
					kernel_regularizer=None),
				tf.keras.layers.Activation('softmax', dtype='float32', name='prediction')
			])
	
	print(model.summary())
	return model


def train_model(model, hparams, train_data_and_size, valid_data_and_size):
	"""Trains model with the given data and hyperparameters.

	Args:
	model: The tf.keras.Model from _build_model().
	hparams: A namedtuple of hyperparameters. This function expects
		.train_epochs: a Python integer with the number of passes over the
		training dataset;
		.learning_rate: a Python float forwarded to the optimizer;
		.momentum: a Python float forwarded to the optimizer;
		.batch_size: a Python integer, the number of examples returned by each
		call to the generators.
	train_data_and_size: A (data, size) tuple in which data is training data to
		be fed in tf.keras.Model.fit(), size is a Python integer with the
		numbers of training.
	valid_data_and_size: A (data, size) tuple in which data is validation data
		to be fed in tf.keras.Model.fit(), size is a Python integer with the
		numbers of validation.

	Returns:
	The tf.keras.callbacks.History object returned by tf.keras.Model.fit().
	"""

	earlystop_callback = tf.keras.callbacks.EarlyStopping(
  		monitor='val_accuracy', min_delta=0.0001,
  		patience=1)

	train_data, train_size = train_data_and_size
	valid_data, valid_size = valid_data_and_size
	# TODO(b/139467904): Expose this hyperparameter as a flag.
	loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=hparams.label_smoothing)

	if hparams.use_mixed_precision is True:
		optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(configure_optimizer(hparams))
	else:
		optimizer = configure_optimizer(hparams)

	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=["accuracy"])
	steps_per_epoch = train_size // hparams.batch_size
	validation_steps = valid_size // hparams.batch_size
	return model.fit(
		train_data,
		use_multiprocessing=False,
		workers=multiprocessing.cpu_count() -1,
		epochs=hparams.train_epochs,
		callbacks=[earlystop_callback],
		steps_per_epoch=steps_per_epoch,
		validation_data=valid_data,
		validation_steps=validation_steps)

def model_to_frozen_graph(model):
 
	# Convert Keras model to ConcreteFunction
	# In the resulting graph, "self" will be the input node
	# and the very last softmax layer in the graph will be the
	# output prediction node.

	full_model = tf.function(model)
	full_model = full_model.get_concrete_function(
	tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

	# Get frozen ConcreteFunction
	frozen_func = convert_variables_to_constants_v2(full_model)
	input_graph = frozen_func.graph.as_graph_def()

	types_to_remove = {"CheckNumerics": True, "Identity": True}

	input_nodes = input_graph.node
	names_to_remove = {}
	
	# We're going to clean up some junk nodes that we do not
	# need outside of training. I assume these are inherited
	# from tensorflow hub.
	for node in input_nodes:
		if '/input_control_node/_'.upper() in node.name.upper():
			names_to_remove[node.name] = True

		if '/output_control_node/_'.upper() in node.name.upper():
			names_to_remove[node.name] = True

	# What we're doing here is double-iterating over the graph nodes
	# looking for disconnected/orphaned nodes. Any node who's name 
	# cannot be found inside the inputs of another node is considered
	# trash that caused me pain and suffering for two days, so they're
	# going to be deleted.
	#
	# On a serious note, these are leftover junk (I assume) from the 
	# tensorflow hub input that is not needed outside of training.
	for node in input_nodes:
		noOutput = True
		for inner in input_nodes:
			resa = [i for i in inner.input if node.name.upper() in i.upper()] 
			if len(resa) > 0:
				noOutput = False

		if noOutput is True:
			names_to_remove[node.name] = True

	# We're going to look for junk nodes (used only in training) that are connected
	# to our output Softmax layer and mark those for deletion as well.
	for node in input_nodes:
		if node.op in types_to_remove:

			# Find all nodes of type Identity that are connected to a Softmax (our output)
			found = [i for i in node.input if 'softmax'.upper() in i.upper()] 

			if found is not None and len(found) > 0:
				names_to_remove[node.name] = True
			
	# The rest of this code is basically a straight-copy-and-paste from
	# the remove_nodes function of TF1.
	nodes_after_removal = []
	for node in input_nodes:
		if node.name in names_to_remove:
			continue
		new_node = node_def_pb2.NodeDef()
		new_node.CopyFrom(node)
		input_before_removal = node.input
		del new_node.input[:]
		for full_input_name in input_before_removal:
			input_name = re.sub(r"^\^", "", full_input_name)
			if input_name in names_to_remove:
				continue
			new_node.input.append(full_input_name)
		nodes_after_removal.append(new_node)

	# TODO - We may be able to just delete all of this code here, as it
	# it was unused by me and I was able to get a functional output.
	# When this TODO is tackled, just delete everything that has to do
	# with node splicing. In the final output, these nodes become either
	# Const or NoOp nodes anyway so they're junk, but harmless junk.
	types_to_splice = {"Identityzzz": True}
	control_input_names = set()
	node_names_with_control_input = set()
	for node in nodes_after_removal:
		for node_input in node.input:
			if "^" in node_input:
				control_input_names.add(node_input.replace("^", ""))
				node_names_with_control_input.add(node.name)

	names_to_splice = {}
	for node in nodes_after_removal:
		if node.op in types_to_splice and node.name not in protected_nodes:
			# We don't want to remove nodes that have control edge inputs, because
			# they might be involved in subtle dependency issues that removing them
			# will jeopardize.
			if node.name not in node_names_with_control_input and len(node.input) > 0:
				names_to_splice[node.name] = node.input[0]

	# We also don't want to remove nodes which are used as control edge inputs.
	names_to_splice = {name: value for name, value in names_to_splice.items()
										 if name not in control_input_names}

	nodes_after_splicing = []
	for node in nodes_after_removal:
		if node.name in names_to_splice:
			continue
		new_node = node_def_pb2.NodeDef()
		new_node.CopyFrom(node)
		input_before_removal = node.input
		del new_node.input[:]
		for full_input_name in input_before_removal:
			input_name = re.sub(r"^\^", "", full_input_name)
			while input_name in names_to_splice:
				full_input_name = names_to_splice[input_name]
				input_name = re.sub(r"^\^", "", full_input_name)
			new_node.input.append(full_input_name)
		nodes_after_splicing.append(new_node)

	output_graph = graph_pb2.GraphDef()
	output_graph.node.extend(nodes_after_splicing)
	return output_graph

def make_image_classifier(tfhub_module, image_dir, hparams,
							requested_image_size=None, saveModelDir=False):
	"""Builds and trains a TensorFLow model for image classification.

	Args:
	tfhub_module: A Python string with the handle of the Hub module.
	image_dir: A Python string naming a directory with subdirectories of images,
		one per class.
	hparams: A HParams object with hyperparameters controlling the training.
	requested_image_size: A Python integer controlling the size of images to
		feed into the Hub module. If the module has a fixed input size, this
		must be omitted or set to that same value.
	"""

	print("Using hparams:")
	for key, value in hparams._asdict().items():
		print("\t{0} : {1}".format(key, value))
		
	module_layer = hub.KerasLayer(tfhub_module, trainable=hparams.do_fine_tuning)
	
	image_size = _image_size_for_module(module_layer, requested_image_size)
	print("Using module {} with image size {}".format(
		tfhub_module, image_size))
	train_data_and_size, valid_data_and_size, labels = _get_data_with_keras(
		image_dir, image_size, hparams.batch_size, hparams.validation_split, hparams.do_data_augmentation)
	print("Found", len(labels), "classes:", ", ".join(labels))

	model = build_model(module_layer, hparams, image_size, len(labels))

	# If we are fine-tuning, check and see if weights
	# already exists at the output directory. This way, a user
	# can simply run two consecutive training sessions. One without
	# fine-tuning, followed by another with.
	if hparams.do_fine_tuning:
		if saveModelDir is not None:
			existingWeightsPath = os.path.join(saveModelDir, "saved_model_weights.h5")
			if os.path.exists(existingWeightsPath):
				print("Loading existing weights for fine-tuning")
				model.load_weights(existingWeightsPath)

	train_result = train_model(model, hparams, train_data_and_size,
							 valid_data_and_size)

	# Tear down model, set training to 0 and then re-create.
	# 1 - Save model weights as Keras H5.

	tempDir = tempfile.gettempdir()
	tempModelWeightsFile = os.path.join(tempDir, "weights.h5")

	model.save_weights(tempModelWeightsFile)

	# 2 - Set training to 0

	K.clear_session()
	K.set_learning_phase(0)

	# 3 - Create model again

	model = build_model(module_layer, hparams, image_size, len(labels))

	# 4 - Load model weights.

	model.load_weights(tempModelWeightsFile)

	# Clean up temp weights file
	os.remove(tempModelWeightsFile)
	
	# 5 - Pass model to lib.model_to_frozen_graph.
	frozen_inference_graph = model_to_frozen_graph(model)

	return model, labels, train_result, frozen_inference_graph
