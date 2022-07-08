# (1) Importing dependency
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

np.random.seed(1000)
''
print("Tensorflow version", tf.__version__)

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
	print("Magic number", magic_number)
	number_of_layers = int.from_bytes(file.read(4), byteorder='big', signed=True)
	print("Number of layers:", number_of_layers)
	for layer_index in range(number_of_layers):
		print("Layer index", layer_index)
		layer_type = int.from_bytes(file.read(4), byteorder='big', signed=True)
		print("Layer type", layer_type)
		if (layer_type == 6):
			type_of_activation = int.from_bytes(file.read(4), byteorder='big', signed=True)
		number_of_weight_indices = int.from_bytes(file.read(4), byteorder='big', signed=True)
		print("Number of weight indices", number_of_weight_indices) #weights and biases
		for weight_index in range(number_of_weight_indices):
			length_of_weights_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
			print("Length of weights shape", length_of_weights_shape)
			weights_shape = []
			for shape_index in range(length_of_weights_shape):
				shape_value = int.from_bytes(file.read(4), byteorder='big', signed=True)
				weights_shape.append(shape_value) 
			number_of_data_stored_in_shape = 1
			print("Weights shape", weights_shape)
			for i in range(len(weights_shape)):
				number_of_data_stored_in_shape *= weights_shape[i]
			print("Number of data stored in shape", number_of_data_stored_in_shape)
			for i in range(number_of_data_stored_in_shape):
				ba = file.read(4)
				ba = struct.unpack("f", ba)[0]
		length_of_output_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
		print("Length of output shape:", length_of_output_shape)
		for output_shape_index in range(length_of_output_shape):
			dim = int.from_bytes(file.read(4), byteorder='big', signed=True)
			print("Dim:", dim)
	file.close()


# (2) Get Data
'''
dataset = tfds.load('stanford_dogs', split='train[:75%]', as_supervised=True)
print(dataset)
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train[:500] #np.reshape(x_train, newshape=(1000, 32, 32, 3))
y_train = y_train[:500]
'''
data_upscaled = np.zeros(shape=(1, 224, 224, 3))
for i, img in enumerate(x_train):
	im = np.transpose(img, (1, 2, 0))
	large_img = cv2.resize(im, dsize = (224, 224), interpolation=cv2.INTER_CUBIC)
	data_upscaled[i] = np.transpose(large_img, (2, 0, 1))
'''

print("x_train shape[0]:", x_train.shape[0])
print("x_train type:", type(x_train))
print("y`_train type:", type(y_train))
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# (3) Create a sequential model
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
validation_split=0.2, shuffle=True)
print(model.summary())

'''print("Evaluating on dog")
im = Image.open('/Users/bora/Desktop/dog.jpg')
im = im.resize((224, 224))
im = np.array(im)
im = im.astype('float32')
im = im / 255
im = np.expand_dims(im, axis=0)
pred = model.predict(im)
'''
'''
print("Saving network:")
save_network(model, "alexnet.nn")
'''
print("Load network:")
load_network(model)

''
labels = open('/Users/bora/Desktop/cifar10_labels.json')
labels = json.load(labels)
print(labels)

print("Evaluating on dog")
im = Image.open('/Users/bora/Desktop/dog.jpg')
im = im.resize((224, 224))
im = np.array(im)
im = im.astype('float32')
im = im / 255
im = np.expand_dims(im, axis=0)
pred = model.predict(im)
#pred = labels["label_names"][np.argmax(pred)]

highest_pred = 0
highest_pred_index = -1
for i in range(len(pred[0])):
	if pred[0][i] > highest_pred:
		highest_pred = pred[0][i]
		highest_pred_index = i
print("Prediction:", labels[highest_pred_index], highest_pred)
