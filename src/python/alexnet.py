# (1) Importing dependency
from base64 import decode
import imp
from operator import le
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D, Resizing
from keras.layers import BatchNormalization
import numpy as np
import tensorflow as tf
import struct
from PIL import Image
import json
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import random

np.random.seed(1000)

def save_network(model, network_file_name):
	file = open(network_file_name, "wb")
	magic_number = 1234
	file.write(magic_number.to_bytes(4, byteorder='big', signed=True))
	print("Magic number", magic_number)
	number_of_layers = len(model.layers)
	print("Number of layers", number_of_layers)
	file.write(number_of_layers.to_bytes(4, byteorder='big', signed=True))
	for layer in model.layers:
		print("Saving layer:", str(layer.name))
		if ('resizing' in str(layer.name)):
			layer_type = 0
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('conv' in str(layer.name)):
			layer_type = 1
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('pool' in str(layer.name)):
			layer_type = 2
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('flatten' in str(layer.name)):
			layer_type = 3
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('dense' in str(layer.name)):
			layer_type = 4
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('batch' in str(layer.name)):
			layer_type = 5
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		elif ('activation' in str(layer.name)):
			layer_type = 6
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
			if ('relu' in str(layer.activation)):
				type_of_activation = 1
				file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
				print("Type of activation", type_of_activation)
			elif ('softmax' in str(layer.activation)):
				type_of_activation = 2
				file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
				print("Type of activation", type_of_activation)
		elif ('dropout' in str(layer.name)):
			layer_type = 7
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			print("Layer type", layer_type)
		number_of_weight_indices = len(layer.get_weights())
		print("Number of weight indices", number_of_weight_indices)
		file.write(number_of_weight_indices.to_bytes(4, byteorder='big', signed=True))
		for weight_index in range(len(layer.get_weights())):
			weightsnumpy = layer.get_weights()[weight_index]
			#print("Weightsnumpy", weightsnumpy)
			length_of_weights_shape = len(weightsnumpy.shape)
			print("Length of weights shape:", length_of_weights_shape)
			file.write(length_of_weights_shape.to_bytes(4, byteorder='big', signed=True))
			for i in range(len(weightsnumpy.shape)):
				print("Shape", weightsnumpy.shape[i])
				file.write(weightsnumpy.shape[i].to_bytes(4, byteorder='big', signed=True))
			flattened_weights = weightsnumpy.flatten()
			for i in range(len(flattened_weights)):
				flattened_weights_value = flattened_weights[i]
				ba = bytearray(struct.pack("f", flattened_weights_value))
				file.write(ba)
		output_shape = layer.output_shape if type(layer.output_shape) is tuple else layer.output_shape[0]
		output_shape = list(output_shape)
		output_shape[0] = 1 # replace None value
		print("Output shape:", output_shape)
		print("Length of output shape:", len(output_shape))
		file.write(len(output_shape).to_bytes(4, byteorder='big', signed=True))
		for dim in output_shape:
			file.write(int(dim).to_bytes(4, byteorder='big', signed=True))
			print("Dim", dim)
	file.close()
	print("Network saved to", network_file_name)

def load_network(model):
	file = open("alexnet.nn", "rb")
	magic_number = int.from_bytes(file.read(4), byteorder='big', signed=True)
	#print("Magic number", magic_number)
	number_of_layers = int.from_bytes(file.read(4), byteorder='big', signed=True)
	#print("Number of layers:", number_of_layers)
	for layer_index in range(number_of_layers):
		#print("Layer index", layer_index)
		layer_type = int.from_bytes(file.read(4), byteorder='big', signed=True)
		#print("Layer type", layer_type)
		if (layer_type == 6):
			type_of_activation = int.from_bytes(file.read(4), byteorder='big', signed=True)
		number_of_weight_indices = int.from_bytes(file.read(4), byteorder='big', signed=True)
		#print("Number of weight indices", number_of_weight_indices) #weights and biases
		nd_list = []
		for weight_index in range(number_of_weight_indices):
			length_of_weights_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
			#print("Length of weights shape", length_of_weights_shape)
			weights_shape = []
			for shape_index in range(length_of_weights_shape):
				shape_value = int.from_bytes(file.read(4), byteorder='big', signed=True)
				weights_shape.append(shape_value)
			nd_array = np.empty(shape=tuple(weights_shape))
			nd_list.append(nd_array)
			number_of_data_stored_in_shape = 1
			#print("Weights shape", weights_shape)
			for i in range(len(weights_shape)):
				number_of_data_stored_in_shape *= weights_shape[i]
			#print("Number of data stored in shape", number_of_data_stored_in_shape)
			it = np.ndindex(nd_array.shape)
			for i in range(number_of_data_stored_in_shape):
				ba = file.read(4)
				ba = struct.unpack("f", ba)[0]
				index = next(it) 
				nd_array[index] = ba
		model.layers[layer_index].set_weights(nd_list)
		length_of_output_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
		#print("Length of output shape:", length_of_output_shape)
		for output_shape_index in range(length_of_output_shape):
			dim = int.from_bytes(file.read(4), byteorder='big', signed=True)
			#print("Dim:", dim)
	file.close()

def create_alexnet():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	x_train = x_train[:1] #np.reshape(x_train, newshape=(1000, 32, 32, 3))
	y_train = y_train[:1]

	model = Sequential()

	# 1st Convolutional Layer
	model.add(Resizing(224, 224, interpolation='bilinear'))
	model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11,11), strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
	# Pooling 
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation before passing it to the next layer
	model.add(BatchNormalization())

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	# Batch Normalisation
	model.add(BatchNormalization())

	# Passing it to a dense layer
	model.add(Flatten())
	# 1st Dense Layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	model.add(Dropout(0.4))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 2nd Dense Layer
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	# Batch Normalisation
	model.add(BatchNormalization())

	# 3rd Dense Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	# Batch Normalisation
	model.add(BatchNormalization())

	# Output Layer
	model.add(Dense(17))
	model.add(Activation('softmax'))


	# (4) Compile 
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',\
	metrics=['accuracy'])



	# (5) Train
	model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, \
	validation_split=0, shuffle=True)

	load_network(model)

	return model

def relu(X):
   return np.maximum(0,X)


def softmax(X):
	expo = np.exp(X)
	expo_sum = np.sum(np.exp(X))
	return expo/expo_sum


def resizing(layer, inputs):
	print("Resizing")
	outputs = layer(inputs)
	return outputs

def conv_2D(layer, inputs):
	print("Conv 2D")
	print('Applying convolution') 
	epsilon = 0.01 # Accepted error 
	max_iters = 1000000 # How many iterations it takes to calculate for ~10 seconds, modify as needed
	
	# Actual output 
	outputs = layer(inputs) 
	outputnp = outputs.numpy()
	
	# Probability of replacing an actual output
	iters = outputs.shape[1] * outputs.shape[2] * outputs.shape[3] * inputs.shape[3] 
	probability = max_iters / iters
	print('Probability:', str(round(probability, 3))) 
	
	# Add padding so output is the same size as input 
	inputs = np.pad(inputs, [(0, 0), (1, 1), (1, 1), (0, 0)], mode='constant') # count, x, y, channel
	
	kernel = layer.kernel.numpy() 
	weights = layer.weights[1].numpy() 
	
	counter = 0 
	prev_value = 0
	for x in range(outputs.shape[1]):
		# Progress indicator 
		if int(x / outputs.shape[1] * 10) != prev_value: 
			prev_value = int(x / outputs.shape[1] * 10) 
			print(str(prev_value * 10) + '%')
		
		for y in range(outputs.shape[2]):
			result = 0 
			for filter_index in range(outputs.shape[3]): 
				# Chance of replacing expected value with custom calculated value 
				if random.random() < probability: 
					result = weights[filter_index]
					for kernel_x in range(layer.kernel_size[0]): 
						for kernel_y in range(layer.kernel_size[1]): 
							for channel in range(inputs.shape[3]): 
								result += kernel[kernel_x][kernel_y][channel][filter_index] * inputs[0][x + kernel_x][y + kernel_y][channel]
					if result < 0: result = 0 
					
					# Compare expected vs actual value, exit if difference is too large 
					expected = outputnp[0][x][y][filter_index] 
					diff = abs(result - expected) 
					if diff > epsilon:
						print('Convolution incorrect!', result, expected, diff)
						sys.exit(0) 
						
					# Replace output with our own calculated value
					outputnp[0][x][y][filter_index] = result
					counter += 1
	
	max_items = outputs.shape[1] * outputs.shape[2] * outputs.shape[3]
	print(str(counter) + '/' + str(max_items), 'items replaced (' + str(round(counter / max_items * 100, 1)) + '%)')  
	
	# Replace with required data type 
	eager_tensor = tf.convert_to_tensor(outputnp, dtype=np.float32)
	return eager_tensor
	outputs = layer(inputs)
	return outputs

def batch_normalization(layer, inputs):
	print("Batch normalization")
	outputs = layer(inputs)
	return outputs

def activation(layer, inputs):
	print("Activation")
	outputs = layer(inputs)
	return outputs

def max_pooling(layer, inputs):
	print("Max pooling")
	outputs = layer(inputs)
	return outputs

def flatten(layer, inputs):
	print("Flatten")
	outputs = layer(inputs)
	return outputs

def dense(layer, inputs):
	print("Dense")
	outputs = layer(inputs)
	return outputs

def dropout(layer, inputs):
	print("Dropout")
	outputs = layer(inputs)
	return outputs


def decode_predictions(pred):
	labels = open('/Users/bora/Desktop/cifar10_labels.json')
	labels = json.load(labels)
	highest_pred = 0
	highest_pred_index = -1
	for i in range(len(pred[0])):
		if pred[0][i] > highest_pred:
			highest_pred = pred[0][i]
			highest_pred_index = i
	print("Prediction:", labels[highest_pred_index], highest_pred)


if __name__ == '__main__':
	model = create_alexnet()

	im = Image.open('/Users/bora/Desktop/dog.jpg')
	test_file = open('/Users/bora/Desktop/test.txt', 'w')
	size = im.size
	image_input = np.zeros(shape=(1, 224, 224, 3)) 
	for x in range(224):
		for y in range(224): 
			pixel = im.getpixel((x / 224 * size[0], y / 224 * size[1])) 
			for p in range(3): 
				image_input[0][y][x][p] = pixel[p] / 255.0
				test_file.write(str(image_input[0][y][x][p]) + ' ')
	# Feedforward
	x = model.layers[0](image_input) # Setting inputs
	for layer in model.layers[1:]:   # Feed forward each layer 
		name = layer.__class__.__name__
		if 'Conv2D' in name:         x = conv_2D(layer, x)
		elif 'MaxPooling2D' in name: x = max_pooling(layer, x)
		elif 'Flatten' in name:      x = flatten(layer, x) 
		elif 'Dense' in name:        x = dense(layer, x)
		elif 'Dropout' in name:      x = dropout(layer, x)
		elif 'BatchNormalization' in name: x = batch_normalization(layer, x)
		elif 'Activation' in name:         x = activation(layer, x)
		elif 'Resizing' in name:           continue
		else: 
			print('Unrecognized layer:', name)
			sys.exit(0)

	pred = decode_predictions(x.numpy()) #Prints the highest predicted class
