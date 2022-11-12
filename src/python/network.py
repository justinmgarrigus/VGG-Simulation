# (1) Importing dependency
import sys
import keras
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.utils.np_utils import to_categorical 
import numpy as np
import tensorflow as tf
import struct
from PIL import Image
import pandas
import random
import math 
import os 
import timer 
from tqdm import tqdm 

np.random.seed(1000)

use_tensorcore = True # Note: doesn't actually use tensorcores, just does matrix multiplication instead 
override_probability = False # If true, the entire calculation is performed for each layer; if false, a fraction of the calculation is performed.
check_accuracy = False # If true, calculated values are compared with the values we should have calculated. 

magic_number = 1234 
layer_types = {'InputLayer': 0, 'Conv2D': 1, 'MaxPooling2D': 2, 'Flatten': 3, 'Dense': 4, 'BatchNormalization': 5} 
activation_types = {'relu': 1, 'softmax': 2}
filtered_layers = {'Dropout'} 
network_types = ['vgg16', 'alexnet', 'lenet'] 


# Returns the sum of every element in an ndarray 
def entropy(arr): 
	total = 0 
	for x in np.nditer(arr): 
		total += x 
	return total 


# Replaces None elements in the input tuple with 1s. 
def none_tuple_replace(tup): 
	if isinstance(tup[0], tuple):
		tup = tup[0] 
		
	lst = list(tup) 
	for i in range(len(lst)):
		if lst[i] is None: 
			lst[i] = 1
	return tuple(lst) 


def save_network(model, file_name):
	print('Saving network to:', file_name)
	file = open(file_name, "wb")
	
	def write_int(value): 
		file.write(value.to_bytes(4, byteorder='big', signed=True)) 
	
	def write_float(value): 
		byte_arr = bytearray(struct.pack("f", float(value))) 
		file.write(byte_arr)
		
	def write_shape(shape): 
		if isinstance(shape, list): 
			shape = shape[0] 
		shape = none_tuple_replace(shape) 
		
		write_int(len(shape)) 
		for dim in shape:
			write_int(dim) 
	
	write_int(1234)
	
	layer_count = len(model.layers) - len([x for x in model.layers if x.__class__.__name__ in filtered_layers]) + (1 if model.layers[0].__class__.__name__ != 'InputLayer' else 0)
	write_int(layer_count) 

	if model.layers[0].__class__.__name__ != 'InputLayer': 
		write_int(0) # layer type  
		write_int(0) # how many weights 
		write_shape(model.layers[0].input_shape) 
	
	for layer in model.layers:
		name = layer.__class__.__name__
		if name in filtered_layers: continue 
		layer_type = layer_types.get(name)
		write_int(layer_type)
		
		if layer_type == layer_types['Conv2D'] or layer_type == layer_types['Dense']: 
			activation_type = activation_types.get(layer.activation.__name__)
			write_int(activation_type) 
		
		if layer_type == layer_types['MaxPooling2D']:
			weights = [np.empty(shape=(1,)), np.empty(shape=(1,))]
			weights[0][0] = layer.strides[0] 
			weights[1][0] = layer.pool_size[0]
		elif layer_type == layer_types['BatchNormalization']: 
			weights = layer.weights
			weights.append(np.array([layer.epsilon])) 
		else: 
			weights = layer.weights 
			if layer_type == layer_types['Conv2D']:
				pad = [0, 1, 1, 0] if layer.padding == 'same' else [0, 0, 0, 0] 
				weights.append(np.array(pad)) 
				weights.append(np.array([layer.strides[0]]))
		
		write_int(len(weights)) 
		weight_counter = 0 
		for weights in weights: 
			write_shape(weights.shape) 
			for weight in np.nditer(weights): 
				write_float(weight)
			weight_counter += 1 
		
		write_shape(layer.output_shape) 
		
	file.close()
	print("Network saved to", file_name)


def load_network(model, file_name):
	print('Loading network from:', file_name) 
	file = open(file_name, "rb")
	
	def read_int(): 
		return int.from_bytes(file.read(4), byteorder='big', signed=True) 
		
	def read_float(): 
		byte_arr = file.read(4)
		return struct.unpack("f", byte_arr)[0]
	
	magic_number = read_int() 
	if magic_number != 1234:
		print('Error: magic number in file', file_name, 'not 1234! Read:', magic_number) 
		sys.exit(1) 
	
	if model.layers[0].__class__.__name__ == 'InputLayer': 
		write_int(0) # layer type  
		write_int(0) # how many weights 
		write_shape(model.layers[0].input_shape) 
	
	for layer in range(read_int()): 
		name = model.layers[layer].__class__.__name__
		if name in filtered_layers: continue 
		
		layer_type = read_int() 
		if layer_type == 0 and name != 'InputLayer': 
			read_int() # how many weights
			for i in range(read_int()): read_int() # shape 
			layer_type = read_int() 
			
		if layer_type == layer_types['Conv2D'] or layer_type == layer_types['Dense']:
			read_int()
		
		weight_set = [] 
		for weights in range(read_int()): 
			shape = []
			for dim in range(read_int()): 
				shape.append(read_int()) 
			
			arr = np.empty(shape=tuple(shape))
			weight_set.append(arr) 
			
			for index in np.ndindex(arr.shape): 
				arr[index] = read_float() 
		
		clipped_weights = weight_set[:len(model.layers[layer].get_weights())]
		model.layers[layer].set_weights(clipped_weights) 
		for out in range(read_int()):
			read_int()
	
	file.close() 
	print('Network loaded') 
	

def create_alexnet(train=False):
	model = keras.models.Sequential([\
	keras.layers.InputLayer(input_shape=(227,227,3), name='beginning'),\
	keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),\
    keras.layers.BatchNormalization(),\
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),\
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),\
    keras.layers.BatchNormalization(),\
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),\
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),\
    keras.layers.BatchNormalization(),\
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),\
    keras.layers.BatchNormalization(),\
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),\
    keras.layers.BatchNormalization(),\
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),\
    keras.layers.Flatten(),\
    keras.layers.Dense(4096, activation='relu'),\
    keras.layers.Dropout(0.5),\
    keras.layers.Dense(4096, activation='relu'),\
    keras.layers.Dropout(0.5),\
    keras.layers.Dense(10, activation='softmax')])
	
	model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
	model.summary()
	
	def process_images(image, label):
		# Normalize images to have a mean of 0 and standard deviation of 1
		image = tf.image.per_image_standardization(image)
		# Resize images from 32x32 to 227x227
		image = tf.image.resize(image, (227,227))
		return image, label
	
	dataset_percent = 0.1 
	validation_percent = 0.1 
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
	train_images = train_images[:int(len(train_images) * dataset_percent)]
	train_labels = train_labels[:int(len(train_labels) * dataset_percent)]
	test_images = test_images[:int(len(test_images) * dataset_percent)]
	test_labels = test_labels[:int(len(test_labels) * dataset_percent)]
	validation_images, validation_labels = train_images[:int(len(train_images) * validation_percent)], train_labels[:int(len(train_labels) * validation_percent)]
	train_images, train_labels = train_images[int(len(train_images) * validation_percent):], train_labels[int(len(train_labels) * validation_percent):]
	train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
	validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
	train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
	train_ds = (train_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	test_ds = (test_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	validation_ds = (validation_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	
	if train or not os.path.exists('data/alexnet.nn'):
		model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1) 
		save_network(model, 'data/alexnet.nn') 
		model.save('data/alexnet-keras')
	else: 
		# load_network(model, 'data/alexnet.nn') # TODO: figure out how to fix this 
		model = keras.models.load_model('data/alexnet-keras') 
		(n_train_images, n_train_labels), (n_test_images, n_test_labels) = tf.keras.datasets.cifar10.load_data()
		
		correct = [] 
		for i in range(10): correct.append([]) 
		incorrect = [] 
		for i in range(10): incorrect.append([]) 
		print(len(correct), len(correct[0])) 
		for i in tqdm(range(len(n_train_images))): 
			image = n_train_images[i] 
			image = tf.image.per_image_standardization(image) 
			image = tf.image.resize(image, (227, 227))
			image = image[np.newaxis, :] 
			prediction = model(image)[0]
			max_p = 0 
			for p in range(len(prediction)): 
				if prediction[p] > prediction[max_p]:
					max_p = p
			if max_p == n_train_labels[i] and prediction[max_p] > 0.95: 
				correct[max_p].append(n_train_images[i])
			else: 
				incorrect[max_p].append(n_train_images[i]) 
		
		print(len(correct), len(incorrect)) 
		for i in range(10): 
			image = correct[i][int(random.random() * len(correct[i]))]
			pilIm = Image.fromarray(image) 
			pilIm.save(str(i) + '.jpg') 
			
		
	model.evaluate(test_ds) 
	
	return model


def create_vgg16(train=False): 
	model = VGG16(weights='imagenet')
	if train or not os.path.exists('data/vgg.nn'): 
		save_network(model, 'data/vgg.nn')
	return model 


def create_lenet(train=False): 
	train = train or not os.path.exists('data/lenet.nn')

	train_set = pandas.read_csv('data/lenet_train.csv') 
	test_set = pandas.read_csv('data/lenet_test.csv') 
	
	Y_train = train_set[['label']] 
	X_train = train_set.drop(train_set.columns[[0]], axis=1) 
	X_test = test_set 
	
	X_train = np.array(X_train) 
	X_test = np.array(X_test) 
	
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) 
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) 
	
	# Padding the images by 2 pixels since the paper input images were 32x32 
	X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant') 
	X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant') 
	
	# Standardization 
	mean_px = X_train.mean().astype(np.float32)
	std_px = X_train.std().astype(np.float32) 
	X_train = (X_train - mean_px)/(std_px) 
	
	Y_train = to_categorical(Y_train)
	
	model = keras.models.Sequential([\
	keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32,32,1)),\
	keras.layers.MaxPool2D(pool_size=2, strides=2),\
	keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14,14,6)),\
	keras.layers.MaxPool2D(pool_size=2, strides=2),\
	keras.layers.Flatten(),\
	keras.layers.Dense(units=120, activation='relu'),\
	keras.layers.Dense(units=84, activation='relu'),\
	keras.layers.Dense(units=10, activation='softmax')]) 
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
	
	epoch_count = 42 if train else 1
	model.fit(X_train, Y_train, steps_per_epoch=10, epochs=epoch_count)
	
	if train: save_network(model, 'data/lenet.nn')
	else: load_network(model, 'data/lenet.nn') 
	
	return model 


def image_mean_dev(image): 
	size = image.size[0] * image.size[1] * 3
	
	# Average of all colors in image 
	elem_sum = 0 
	for x in range(image.size[0]):
		for y in range(image.size[1]): 
			pixel = image.getpixel((x, y)) 
			if isinstance(pixel, int): elem_sum += pixel 
			else: 
				for p in pixel:
					elem_sum += p 
	mean = elem_sum / size

	# Standard deviation
	deviation_sum = 0 
	for x in range(image.size[0]):
		for y in range(image.size[1]): 
			pixel = image.getpixel((x, y)) 
			if isinstance(pixel, int): deviation_sum += (pixel - mean) ** 2 
			else: 
				for p in pixel: 
					deviation_sum += (p - mean) ** 2
	dev = math.sqrt(deviation_sum / size)
	adjusted_dev = max(dev, 1.0 / math.sqrt(size)) 
	
	return (mean, adjusted_dev) 


def alexnet_preprocess(image, input_shape):
	mean, dev = image_mean_dev(image) 
	
	image_input = np.empty(shape=input_shape)
	for x in range(input_shape[1]): 
		im_x = int(x / input_shape[1] * image.size[0])
		for y in range(input_shape[2]):
			im_y = int(y / input_shape[2] * image.size[1]) 
			pixel = image.getpixel((im_x, im_y))
			for p in range(3): 
				image_input[0][y][x][p] = (pixel[p] - mean) / dev
	
	print(f'Mean: {mean}, dev: {dev}, entropy: {entropy(image_input)}') 
	return image_input 


def vgg16_preprocess(image, input_shape): 
	# Preprocessing: getting the average R, G, and B values
	avg = [0, 0, 0]  
	for x in range(image.size[0]): 
		for y in range(image.size[1]): 
			pixel = image.getpixel((x, y)) 
			for p in range(3): 
				avg[p] += pixel[p] 
	for p in range(3): 
		avg[p] /= image.size[0] * image.size[1] 
		avg[p] = int(avg[p])

	image_input = np.empty(shape=input_shape) 
	for x in range(input_shape[1]): 
		im_x = int(x / input_shape[1] * image.size[0])
		for y in range(input_shape[2]):
			im_y = int(y / input_shape[2] * image.size[1]) 
			pixel = image.getpixel((im_x, im_y))
			for p in range(3): 
				image_input[0][y][x][p] = pixel[p] - avg[p]
				
			# Flip R and B 
			temp = image_input[0][y][x][2] 
			image_input[0][y][x][2] = image_input[0][y][x][0] 
			image_input[0][y][x][0] = temp 
	
	return image_input 


def lenet_preprocess(image, input_shape): 
	mean, dev = image_mean_dev(image) 

	image_input = np.zeros(shape=input_shape) 
	for x in range(image.size[0]):
		for y in range(image.size[1]): 	
			pixel = image.getpixel((x, y))
			image_input[0][x + 2][y + 2] = (pixel - mean) / dev

	return image_input 
	

def relu(X):
   return np.maximum(0,X)


def softmax(X):
	expo = np.exp(X)
	expo_sum = np.sum(np.exp(X))
	return expo/expo_sum


def conv_2D(layer, inputs):
	print('Conv 2D ', end='', flush=True) 
	
	outputnp = conv_2D_tensorcore(layer, inputs) if use_tensorcore else conv_2D_normal(layer, inputs)
	
	# Replace with required data type 
	eager_tensor = tf.convert_to_tensor(outputnp, dtype=np.float32)
	return eager_tensor


def conv_2D_normal(layer, inputs): 
	epsilon = 0.01 # Accepted error  
	
	# Actual output
	outputs = layer(inputs) 
	outputnp = outputs.numpy()
	inputsnp = inputs if isinstance(inputs, np.ndarray) else inputs.numpy()
	
	# Probability of replacing an actual output
	if not override_probability:
		iters_per_second = 1000 
		total_seconds = 60 
		total_iters = iters_per_second * total_seconds
		iters = outputs.shape[1] * outputs.shape[2] * outputs.shape[3] * inputsnp.shape[3] * layer.kernel_size[0] * layer.kernel_size[1] 
		probability = total_iters / iters
	
	# Add padding so output is the same size as input 
	if layer.padding == 'same': 
		pad = (layer.kernel_size[0] - 1) // 2
		inputsnp = np.pad(inputsnp, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant') # count, x, y, channel
	
	kernel = layer.kernel.numpy() 
	weights = layer.weights[1].numpy() 
	kernel_size = (layer.kernel_size[0], layer.kernel_size[1]) 
	strides = (layer.strides[0], layer.strides[1]) 
	
	timer.start() 
	
	dot_counter = 0 
	resultnp = np.empty(shape=outputs.shape) 
	for x in range(outputnp.shape[1]):
		for y in range(outputnp.shape[2]):
			result = 0 
			for filter_index in range(outputnp.shape[3]): 
				# Chance of replacing expected value with custom calculated value
				dot_counter += 1 
				if override_probability or random.random() < probability: 
					result = weights[filter_index]
					for kernel_x in range(kernel_size[0]): 
						for kernel_y in range(kernel_size[1]): 
							for channel in range(inputsnp.shape[3]): 
								result += kernel[kernel_x][kernel_y][channel][filter_index] * inputsnp[0][x * strides[0] + kernel_x][y * strides[1] + kernel_y][channel]
					if result < 0: result = 0
						
					# Replace output with our own calculated value
					resultnp[0][x][y][filter_index] = result
				else: 
					resultnp[0][x][y][filter_index] = outputnp[0][x][y][filter_index] 
	
	np.testing.assert_array_almost_equal(outputnp, resultnp, decimal=1)
	
	timer.stop() 
	print(timer.elapsed()) 
	print('  Dot products:', dot_counter) 
	return outputnp
	

def conv_2D_tensorcore(layer, inputs): 
	assert(layer.strides[0] == layer.strides[1]) 
	outputs = layer(inputs) # actual output we should compare against 
	outputnp = outputs.numpy()
	
	inputsnp = inputs if isinstance(inputs, np.ndarray) else inputs.numpy()
	kernel = layer.kernel.numpy()
	weights = layer.weights[1].numpy() 
	
	print() 
	print('  Info:') 
	print('    stride:', layer.strides[0]) 
	print('    input shape:', inputsnp.shape) 
	print('    output shape:', outputnp.shape) 
	print('    kernel shape:', kernel.shape) 
	
	if layer.padding == 'same': 
		pad = (layer.kernel_size[0] - 1) // 2
		inputsnp = np.pad(inputsnp, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant') # count, x, y, channel
		
	input_adjusted = img2col(inputsnp, kernel.shape, layer.strides[0]) 
	kernel_adjusted = kernel2col(kernel) 
	
	result_col = np.matmul(input_adjusted, kernel_adjusted)
	for filter_index in range(weights.shape[0]): 
		for spot in range(result_col.shape[0]):
			result_col[spot][filter_index] += weights[filter_index] 
			if result_col[spot][filter_index] < 0:
				result_col[spot][filter_index] = 0 
	
	resultnp = col2img(result_col) 
	
	np.testing.assert_array_almost_equal(outputnp, resultnp, decimal=1)
	
	print('  Dot products:', input_adjusted.shape[0] * kernel_adjusted.shape[1])
	print('  Entropy:') 
	print('    input_adjusted:', entropy(input_adjusted)) 
	print('    result_col:', entropy(result_col)) 
	print('    resultnp:', entropy(resultnp)) 
	
	return resultnp
	

# Converts a 4D array into a 2D matrix. 
# arr: must be of size (1, X, X, X)
# kernel_shape: (X, X, X, X)  
# padding_amount: int 
# stride_amount: int 
def img2col(arr, kernel_shape, stride_amount): 
	assert len(arr.shape) == 4 
	assert arr.shape[0] == 1 
	assert len(kernel_shape) == 4 
	assert kernel_shape[0] == kernel_shape[1] # Assumes that kernels are square

	# Warning: a potential source of error is the orientation of the image. This may reflect
	# the image across the wrong access, so check this first. 
	kernel_length = kernel_shape[0]
	width = math.floor(((arr.shape[1] - kernel_length) / stride_amount) + 1)
	height = math.floor(((arr.shape[2] - kernel_length) / stride_amount) + 1) 
	cols = np.zeros(shape=(width * height, arr.shape[3] * kernel_length * kernel_length)) # Note: I reversed the row and col sizes
	
	print('  cols shape:', cols.shape) 
	
	for dot in range(cols.shape[0]): # How many dot products we are calculating 
		col = np.zeros(cols.shape[1])  # The number of multiplications in one dot product
		col_index = 0 
		x = dot // height * stride_amount 
		y = dot % width * stride_amount
		for i in np.ndindex((kernel_shape[2], kernel_shape[0], kernel_shape[1])):
			col[col_index] = arr[0, x + i[1], y + i[2], i[0]] 
			col_index += 1 
		cols[dot] = col 
	return cols 


def kernel2col(kernel): 
  col = np.zeros(shape=(kernel.shape[0] * kernel.shape[1] * kernel.shape[2], kernel.shape[3]))
  for c in range(col.shape[1]):
    counter = 0 
    for i in np.ndindex((kernel.shape[2], kernel.shape[0], kernel.shape[1])):
      col[counter, c] = kernel[i[1], i[2], i[0], c] 
      counter += 1 
  return col 
  
  
def col2img(arr): 
  length = int(math.sqrt(arr.shape[0])) # Assuming shape is a square 
  resultnp = np.zeros(shape=(1, length, length, arr.shape[1]))
  for filter_index in range(arr.shape[1]): 
    for i in range(arr.shape[0]): 
      resultnp[0, i // length, i % length, filter_index] = arr[i, filter_index] 
  return resultnp 

	
def batch_normalization(layer, inputs):
	print('Batch normalization')
	outputs = layer(inputs)
	outputsnp = outputs.numpy()
	inputsnp = inputs if isinstance(inputs, np.ndarray) else inputs.numpy()
	
	gamma = layer.weights[0].numpy() 
	beta = layer.weights[1].numpy() 
	running_mean = layer.weights[2].numpy() 
	running_std = layer.weights[3].numpy()
	epsilon = layer.epsilon
	
	result = np.empty(shape=inputs.shape) 
	for i in range(inputs.shape[1]): 
		for j in range(inputs.shape[2]): 
			for k in range(inputs.shape[3]): 
				result[0, i, j, k] = gamma[k] * (inputsnp[0, i, j, k] - running_mean[k]) / math.sqrt(running_std[k] + epsilon) + beta[k]
			
	np.testing.assert_array_almost_equal(outputsnp, result, decimal=1)
	
	print('  Entropy:', entropy(outputsnp))
	print('    gamma:', entropy(gamma)) 
	print('    beta:', entropy(beta)) 
	print('    running_mean:', entropy(running_mean)) 
	print('    running_std:', entropy(running_std)) 
	print('    epsilon:', epsilon) 
	return tf.convert_to_tensor(result, dtype=np.float32)


def max_pooling(layer, inputs):
	print('Max pooling') 
	outputs = layer(inputs)
	inputsnp = inputs.numpy()
	outputsnp = outputs.numpy()
	resultnp = np.empty(shape=outputsnp.shape) 
	
	limit_x = inputsnp.shape[1] - (inputsnp.shape[1] % layer.strides[0]) 
	limit_y = inputsnp.shape[2] - (inputsnp.shape[2] % layer.strides[1]) 
	for offset_x in range (0, limit_x, layer.strides[0]):
		for offset_y in range (0, limit_y, layer.strides[1]):
			for z in range (0, inputsnp.shape[3]):
				max_value = float('-inf');
				for kernel_x in range (layer.pool_size[0]):
					for kernel_y in range (layer.pool_size[1]):
						kx = offset_x + kernel_x 
						ky = offset_y + kernel_y
						if inputsnp[0][kx][ky][z] > max_value: 
							max_value = inputsnp[0][kx][ky][z]
				x = offset_x // layer.strides[0]
				y = offset_y // layer.strides[1]
				resultnp[0][x][y][z] = max_value 

	np.testing.assert_array_almost_equal(outputsnp, resultnp, decimal=1)
	
	print('  Entropy:', entropy(resultnp)) 
	
	eager_tensor = tf.convert_to_tensor(resultnp, dtype=np.float32)
	return eager_tensor


def flatten(layer, inputs):
	print('Flatten ', end='', flush=True) 
	result_array = np.empty(shape=none_tuple_replace(layer.output_shape))  
	
	timer.start() 
	
	i = 0
	for x in np.nditer(inputs.numpy()): 
		result_array[0][i] = x
		i += 1
	
	timer.stop() 
	print(timer.elapsed()) 
	
	eager_tensor = tf.convert_to_tensor(result_array, dtype=np.float32)
	return eager_tensor 


def dense(layer, inputs):
	print('Dense')
	outputsnp = layer(inputs).numpy() 
	inputsnp = inputs.numpy()
	weightsnp = layer.get_weights()[0]
	biasnp = layer.get_weights()[1] 
	
	if use_tensorcore: 
		result_array = np.matmul(inputs.numpy(), weightsnp) 
		result_array += biasnp 
	else: 
		result_array = np.reshape(np.copy(biasnp), newshape=(1, biasnp.shape[0]))
		for i in range(len(inputsnp)):
			for j in range(len(weightsnp[0])):
				for k in range(len(weightsnp)):
					result_array[i][j] += inputsnp[i][k] * weightsnp[k][j] 
	
	if ('relu' in str(layer.activation)):
		result_array = relu(result_array)
	elif ('softmax' in str(layer.activation)):
		result_array = softmax(result_array) 
	else: 
		print('Unrecognized activation function:', layer.activation)
		sys.exit(0)   
	
	print('  Entropy:', entropy(result_array))  
	np.testing.assert_array_almost_equal(outputsnp, result_array, decimal=1)
	
	eager_tensor = tf.convert_to_tensor(result_array, dtype=np.float32)
	return eager_tensor


def dropout(layer, inputs):
	# Note: 'dropout' is something that is only used in training, it's not used regularly! 
	return tf.convert_to_tensor(inputs, dtype=np.float32)
	

def top_prediction(pred, model_type):
	print('Model:') 
	for i in range(len(pred[0])):
		print(' ', pred[0][i])
	
	if model_type == 'vgg16':
		p = decode_predictions(pred) 
		print(p[0][0]) 
	elif model_type == 'lenet': 
		highest_pred = 0
		highest_pred_index = -1
		for i in range(len(pred[0])):
			if pred[0][i] > highest_pred:
				highest_pred = pred[0][i]
				highest_pred_index = i
		print(highest_pred_index) 
	else: 
		labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		highest_pred = 0
		highest_pred_index = -1
		for i in range(len(pred[0])):
			if pred[0][i] > highest_pred:
				highest_pred = pred[0][i]
				highest_pred_index = i
		print(labels[highest_pred_index], highest_pred)


def feedforward(model, image): 
	# Feedforward
	x = image
	for layer in model.layers:
		name = layer.__class__.__name__
		if 'Conv2D' in name:         		x = conv_2D(layer, x)
		elif 'MaxPooling2D' in name: 		x = max_pooling(layer, x)
		elif 'Flatten' in name:      		x = flatten(layer, x) 
		elif 'Dense' in name:        		x = dense(layer, x)
		elif 'Dropout' in name:      		x = dropout(layer, x)
		elif 'BatchNormalization' in name: 	x = batch_normalization(layer, x)
		elif 'InputLayer' in name: 			x = layer(x) 
		else: 
			print('Unrecognized layer:', name)
			sys.exit(0)
	
	print('\nTotal elapsed time in feedforward:', round(timer.elapsed(), 2)) 
	
	return x 


def test_image(model, model_type, image_name):
	input_shape = none_tuple_replace(model.layers[0].input_shape)
	im = Image.open(image_name)
	
	if model_type == 'alexnet': preprocess = alexnet_preprocess
	elif model_type == 'vgg16': preprocess = vgg16_preprocess 
	else: preprocess = lenet_preprocess 
	image_input = preprocess(im, input_shape)
	
	actual = model(image_input).numpy() 
	ours = feedforward(model, image_input).numpy() 
	
	print() 
	print('Name:', image_name) 
	print('Actual: ', end='') 
	top_prediction(actual, model_type)
	print('Ours: ', end='') 
	top_prediction(ours, model_type) 
	print() 


if __name__ == '__main__':
	if len(sys.argv) < 2 or len(sys.argv) > 3: 
		print('Usage: python3 src/python/network.py <model> <options>')
		print('  Models: -alexnet -vgg16 -lenet')
		print('  Options: -train') 
		sys.exit(1) 
	
	name = sys.argv[1] 
	if name[0] == '-': name = name[1:]
	if name not in network_types:
		print('Model type', sys.argv[1], 'does not match known model.')
		print('Known models: -alexnet -vgg16 -lenet') 
		sys.exit(1) 
	
	train = (sys.argv[2] == '-train') if len(sys.argv) == 3 else False
	if name == 'alexnet': model = create_alexnet(train)
	elif name == 'vgg16': model = create_vgg16(train) 
	else: model = create_lenet(train) 
	
	model.summary() 
	
	if not train: 
		if name == 'vgg16': test_files = ['dog.jpg'] 
		elif name == 'alexnet': test_files = ['mini_dog.jpg'] # ['mini_dog.jpg', 'mini_horse.jpg', 'mini_car.jpg']
		else: test_files = ['digit.jpg']
		
		for test_file in test_files: 
			test_image(model, name, 'data/' + test_file) 