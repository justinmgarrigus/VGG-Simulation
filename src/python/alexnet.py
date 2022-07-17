# (1) Importing dependency
import sys
import keras
import numpy as np
import tensorflow as tf
import struct
from PIL import Image
import random
import math 

np.random.seed(1000)

def save_network(model, network_file_name):
	print('Saving network to:', network_file_name) 
	file = open(network_file_name, "wb")
	magic_number = 1234
	file.write(magic_number.to_bytes(4, byteorder='big', signed=True))
	number_of_layers = len(model.layers)
	file.write(number_of_layers.to_bytes(4, byteorder='big', signed=True))
	for layer in model.layers:
		if ('resizing' in str(layer.name)):
			layer_type = 0
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('conv' in str(layer.name)):
			layer_type = 1
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('pool' in str(layer.name)):
			layer_type = 2
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('flatten' in str(layer.name)):
			layer_type = 3
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('dense' in str(layer.name)):
			layer_type = 4
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('batch' in str(layer.name)):
			layer_type = 5
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		elif ('activation' in str(layer.name)):
			layer_type = 6
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
			if ('relu' in str(layer.activation)):
				type_of_activation = 1
				file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
			elif ('softmax' in str(layer.activation)):
				type_of_activation = 2
				file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
		elif ('dropout' in str(layer.name)):
			layer_type = 7
			file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
		number_of_weight_indices = len(layer.get_weights())
		file.write(number_of_weight_indices.to_bytes(4, byteorder='big', signed=True))
		for weight_index in range(len(layer.get_weights())):
			weightsnumpy = layer.get_weights()[weight_index]
			length_of_weights_shape = len(weightsnumpy.shape)
			file.write(length_of_weights_shape.to_bytes(4, byteorder='big', signed=True))
			for i in range(len(weightsnumpy.shape)):
				file.write(weightsnumpy.shape[i].to_bytes(4, byteorder='big', signed=True))
			flattened_weights = weightsnumpy.flatten()
			for i in range(len(flattened_weights)):
				flattened_weights_value = flattened_weights[i]
				ba = bytearray(struct.pack("f", flattened_weights_value))
				file.write(ba)
		output_shape = layer.output_shape if type(layer.output_shape) is tuple else layer.output_shape[0]
		output_shape = list(output_shape)
		output_shape[0] = 1 # replace None value
		file.write(len(output_shape).to_bytes(4, byteorder='big', signed=True))
		for dim in output_shape:
			file.write(int(dim).to_bytes(4, byteorder='big', signed=True))
	file.close()
	print("Network saved to", network_file_name)


def load_network(model, file_name='data/alexnet.nn'):
	print('Loading network from:', file_name) 
	file = open(file_name, "rb")
	magic_number = int.from_bytes(file.read(4), byteorder='big', signed=True)
	if magic_number != 1234:
		print('Error: magic number in file', file_name, 'not 1234! Read:', magic_number) 
		sys.exit(1) 
	number_of_layers = int.from_bytes(file.read(4), byteorder='big', signed=True)
	for layer_index in range(number_of_layers):
		layer_type = int.from_bytes(file.read(4), byteorder='big', signed=True)
		if (layer_type == 6):
			file.read(4) # type of activation
		number_of_weight_indices = int.from_bytes(file.read(4), byteorder='big', signed=True)
		nd_list = []
		for weight_index in range(number_of_weight_indices):
			length_of_weights_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
			weights_shape = []
			for shape_index in range(length_of_weights_shape):
				shape_value = int.from_bytes(file.read(4), byteorder='big', signed=True)
				weights_shape.append(shape_value)
			nd_array = np.empty(shape=tuple(weights_shape))
			nd_list.append(nd_array)
			number_of_data_stored_in_shape = 1
			for i in range(len(weights_shape)):
				number_of_data_stored_in_shape *= weights_shape[i]
			it = np.ndindex(nd_array.shape)
			for i in range(number_of_data_stored_in_shape):
				ba = file.read(4)
				ba = struct.unpack("f", ba)[0]
				index = next(it) 
				nd_array[index] = ba
		model.layers[layer_index].set_weights(nd_list)
		length_of_output_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
		for output_shape_index in range(length_of_output_shape):
			file.read(4) # dim
	file.close()
	print("Network loaded")
	

def create_alexnet(train=False):
	model = keras.models.Sequential([\
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
	
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
	validation_images, validation_labels = train_images[:5000], train_labels[:5000]
	train_images, train_labels = train_images[5000:], train_labels[5000:]
	train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
	validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
	train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
	train_ds = (train_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	test_ds = (test_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	validation_ds = (validation_ds.map(process_images).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
	
	if train: 
		model.fit(train_ds, epochs=1, validation_data=validation_ds, validation_freq=1)
		save_network(model, 'data/alexnet.nn')
	else: 
		load_network(model, 'data/alexnet.nn')

	model.evaluate(test_ds) 
	
	return model


def preprocess_image(arr): 
	# Find average of all values in arr 
	elem_sum = 0 
	for i in np.ndindex(arr.shape): 
		elem_sum += float(arr[i])
	mean = elem_sum / arr.size 
	
	# Find standard deviation
	dev_sum = 0 
	for i in np.ndindex(arr.shape): 
		dev_sum += (float(arr[i]) - mean) ** 2
	dev = (dev_sum / arr.size) ** 0.5
	
	adjusted_dev = max(dev, 1.0 / math.sqrt(arr.size)) 
	for i in np.ndindex(arr.shape): 
		arr[i] = (arr[i] - mean) / adjusted_dev


# Replaces None elements in the input tuple with 1s. 
def none_tuple_replace(tup): 
	lst = list(tup) 
	for i in range(len(lst)):
		if lst[i] is None: 
			lst[i] = 1
	return tuple(lst) 


def relu(X):
   return np.maximum(0,X)


def softmax(X):
	expo = np.exp(X)
	expo_sum = np.sum(np.exp(X))
	return expo/expo_sum


def conv_2D(layer, inputs):
	print('Conv 2D') 
	epsilon = 0.01 # Accepted error 
	iters_per_second = 1000 
	total_seconds = 60 
	total_iters = iters_per_second * total_seconds 
	
	# Actual output 
	outputs = layer(inputs) 
	outputnp = outputs.numpy()
	
	# Probability of replacing an actual output
	iters = outputs.shape[1] * outputs.shape[2] * outputs.shape[3] * inputs.shape[3] * layer.kernel_size[0] * layer.kernel_size[1] 
	probability = total_iters / iters
	
	# Add padding so output is the same size as input 
	if layer.padding == 'same': 
		pad = (layer.kernel_size[0] - 1) // 2
		inputs = np.pad(inputs, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode='constant') # count, x, y, channel
	
	kernel = layer.kernel.numpy() 
	weights = layer.weights[1].numpy() 
	
	for x in range(outputs.shape[1]):
		for y in range(outputs.shape[2]):
			result = 0 
			for filter_index in range(outputs.shape[3]): 
				# Chance of replacing expected value with custom calculated value 
				if random.random() < probability: 
					result = weights[filter_index]
					for kernel_x in range(layer.kernel_size[0]): 
						for kernel_y in range(layer.kernel_size[1]): 
							for channel in range(inputs.shape[3]): 
								result += kernel[kernel_x][kernel_y][channel][filter_index] * inputs[0][x * layer.strides[0] + kernel_x][y * layer.strides[1] + kernel_y][channel]
					if result < 0: result = 0 
					
					# Compare expected vs actual value, exit if difference is too large 
					expected = outputnp[0][x][y][filter_index] 
					diff = abs(result - expected) 
					if diff > epsilon:
						print('Convolution incorrect!', result, expected, diff)
						sys.exit(0) 
						
					# Replace output with our own calculated value
					outputnp[0][x][y][filter_index] = result
	
	# Replace with required data type 
	eager_tensor = tf.convert_to_tensor(outputnp, dtype=np.float32)
	return eager_tensor


def batch_normalization(layer, inputs):
	print("Batch normalization")
	outputs = layer(inputs)
	outputsnp = outputs.numpy()
	
	gamma = layer.weights[0].numpy() 
	beta = layer.weights[1].numpy() 
	running_mean = layer.weights[2].numpy() 
	running_std = layer.weights[3].numpy()
	
	probability = (outputsnp.size / inputs.shape[3] * 30) / outputsnp.size 
	
	result = np.empty(shape=inputs.shape) 
	it = np.ndindex(result.shape) 
	for i in range(outputsnp.size // inputs.shape[3]):
		if random.random() < probability: 
			for f in range(inputs.shape[3]): 
				index = next(it)
				result[index] = gamma[f] * (inputs[index] - running_mean[f]) / math.sqrt(running_std[f] + layer.epsilon) + beta[f]
		else: 
			for f in range(inputs.shape[3]): 
				index = next(it) 
				result[index] = outputsnp[index] 
	
	# Verify average difference is less than 0.01 
	diff = 0 
	for x in np.ndindex(result.shape):
		diff += abs(result[x] - outputsnp[x]) 
	error = diff / outputsnp.size / probability
	if error > 0.01: 
		print('BatchNormalization error too high!', error)
		sys.exit(0) 
	
	return tf.convert_to_tensor(result, dtype=np.float32)


def max_pooling(layer, inputs):
	print('Max pooling') 
	outputs = layer(inputs)
	inputsnp = inputs.numpy()
	outputsnp = outputs.numpy()
	
	for offset_x in range (0, inputsnp.shape[1], 2):
		for offset_y in range (0, inputsnp.shape[2], 2):
			for z in range (0, inputsnp.shape[3]):
				max_value = float('-inf')
				for kernel_x in range (2):
					for kernel_y in range (2):
						x = offset_x + kernel_x 
						y = offset_y + kernel_y 
						if inputsnp.shape[1] > x and inputsnp.shape[2] > y: 							
							if inputsnp[0][x][y][z] > max_value:
								max_value = inputsnp[0][x][y][z]
				x = offset_x // 2 
				y = offset_y // 2 
				if outputsnp.shape[1] > x and outputsnp.shape[2] > y: 
					outputsnp[0][x][y][z] = max_value

	eager_tensor = tf.convert_to_tensor(outputsnp, dtype=np.float32)
	return eager_tensor


def flatten(layer, inputs):
	print('Flatten') 
	result_array = np.empty(shape=none_tuple_replace(layer.output_shape)) 
	i = 0 
	for x in np.nditer(inputs.numpy()): 
		result_array[0][i] = x
		i += 1
	eager_tensor = tf.convert_to_tensor(result_array, dtype=np.float32)
	return eager_tensor 


def dense(layer, inputs):
	print('Dense')
	result_array = np.zeros(shape=none_tuple_replace(layer.output_shape))
	inputsnp = inputs.numpy()
	weightsnp = layer.get_weights()[0]

	for i in range(len(inputsnp)):
	# iterating by column by B
		for j in range(len(weightsnp[0])):
			# iterating by rows of B
			for k in range(len(weightsnp)):
				result_array[i][j] += inputsnp[i][k] * weightsnp[k][j]


	# matrix_mul_array = np.matmul(inputs.numpy(), layer.get_weights()[0])
	bias_added_array = result_array + layer.get_weights()[1]
	if ('relu' in str(layer.activation)):
		activation_function_array = relu(bias_added_array)
	elif ('softmax' in str(layer.activation)):
	   activation_function_array = softmax(bias_added_array)
	else: 
		print('Unrecognized activation function:', layer.activation)
		sys.exit(0)   

	eager_tensor = tf.convert_to_tensor(activation_function_array, dtype=np.float32)
	return eager_tensor


def dropout(layer, inputs):
	# Note: 'dropout' is something that is only used in training, it's not used regularly! 
	return tf.convert_to_tensor(inputs, dtype=np.float32)
	

def decode_predictions(pred):
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
		else: 
			print('Unrecognized layer:', name)
			sys.exit(0)
	return x 


def test_image(model, name):
	im = Image.open(name) 
	image_input = np.empty(shape=(1, 227, 227, 3)) 
	for x in range(227): 
		for y in range(227): 
			pixel = im.getpixel((x / 227 * 32, y / 227 * 32))
			for p in range(3): 
				image_input[0][y][x][p] = pixel[p] 
	preprocess_image(image_input) 
	
	actual = model(image_input).numpy() 
	ours = feedforward(model, image_input).numpy() 
	
	print() 
	print('Name:', name) 
	print('Actual: ', end='') 
	decode_predictions(actual)
	print('Ours: ', end='') 
	decode_predictions(ours) 
	print() 


if __name__ == '__main__':
	model = create_alexnet(train=False)

	images = ['dog2.png', 'horse.png', 'car.png'] 
	for name in images: 
		test_image(model, 'data/' + name) 