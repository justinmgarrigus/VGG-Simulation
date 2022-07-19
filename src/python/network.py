# (1) Importing dependency
import sys
import keras
from keras.applications.vgg16 import VGG16, decode_predictions
import numpy as np
import tensorflow as tf
import struct
from PIL import Image
import random
import math 
import os 

np.random.seed(1000)

magic_number = 1234 
layer_types = {'InputLayer': 0, 'Conv2D': 1, 'MaxPooling2D': 2, 'Flatten': 3, 'Dense': 4, 'BatchNormalization': 5} 
activation_types = {'relu': 1, 'softmax': 2}
filtered_layers = {'Dropout'} 


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
		byte_arr = bytearray(struct.pack("f", value)) 
		file.write(byte_arr)
		
	def write_shape(shape): 
		if isinstance(shape, list): 
			shape = shape[0] 
		shape = none_tuple_replace(shape) 
		
		write_int(len(shape)) 
		for dim in shape:
			write_int(dim) 
	
	write_int(1234)
	
	layer_count = len(model.layers) - len([x for x in model.layers if x.__class__.__name__ in filtered_layers]) 
	write_int(layer_count) 
	
	print(len(model.layers), layer_count) 
	
	for layer in model.layers:
		name = layer.__class__.__name__
		if name in filtered_layers: continue 
		layer_type = layer_types.get(name)
		write_int(layer_type) 
		
		if layer_type == layer_types['Conv2D'] or layer_type == layer_types['Dense']: 
			activation_type = activation_types.get(layer.activation.__name__)
			write_int(activation_type) 
		
		write_int(len(layer.weights)) 
		for weights in layer.weights: 
			write_shape(weights.shape) 
			for weight in np.nditer(weights): 
				write_float(weight) 
		
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
	
	for layer in range(read_int()): 
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
		
		model.layers[layer].set_weights(weight_set) 
		for out in range(read_int()):
			read_int()
	
	file.close() 
	print('Network loaded') 
	

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
	
	if train or not os.path.exists('data/alexnet.nn'):
		model.fit(train_ds, epochs=1, validation_data=validation_ds, validation_freq=1) 
		save_network(model, 'data/alexnet.nn') 
	else: 
		load_network(model, 'data/alexnet.nn') 
	
	model.evaluate(test_ds) 
	
	return model


def create_vgg16(train=False): 
	model = VGG16(weights='imagenet')
	if train or not os.path.exists('data/alexnet.nn'): 
		save_network(model, 'data/vgg16.nn')
	return model 


def alexnet_preprocess(image, input_shape):
	size = image.size[0] * image.size[1] * 3
	
	# Average of all colors in image 
	elem_sum = 0 
	for x in range(image.size[0]):
		for y in range(image.size[1]): 
			pixel = image.getpixel((x, y)) 
			for p in pixel:
				elem_sum += p 
	mean = elem_sum / size

	# Standard deviation
	deviation_sum = 0 
	for x in range(image.size[0]):
		for y in range(image.size[1]): 
			pixel = image.getpixel((x, y)) 
			for p in pixel: 
				deviation_sum += (p - mean) ** 2
	dev = math.sqrt(deviation_sum / size)
	adjusted_dev = max(dev, 1.0 / math.sqrt(size)) 
	
	image_input = np.empty(shape=input_shape)
	for x in range(input_shape[1]): 
		im_x = int(x / input_shape[1] * image.size[0])
		for y in range(input_shape[2]):
			im_y = int(y / input_shape[2] * image.size[1]) 
			pixel = image.getpixel((im_x, im_y))
			for p in range(3): 
				image_input[0][y][x][p] = (pixel[p] - mean) / adjusted_dev
	
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
	
	print(avg) 
	return image_input 


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
	inputsnp = inputs.numpy()
	weightsnp = layer.get_weights()[0]
	biasnp = layer.get_weights()[1] 
	
	result_array = np.matmul(inputs.numpy(), weightsnp) 
	result_array += biasnp 
	
	if ('relu' in str(layer.activation)):
		result_array = relu(result_array)
	elif ('softmax' in str(layer.activation)):
		result_array = softmax(result_array) 
	else: 
		print('Unrecognized activation function:', layer.activation)
		sys.exit(0)   

	eager_tensor = tf.convert_to_tensor(result_array, dtype=np.float32)
	return eager_tensor


def dropout(layer, inputs):
	# Note: 'dropout' is something that is only used in training, it's not used regularly! 
	return tf.convert_to_tensor(inputs, dtype=np.float32)
	

def top_prediction(pred, model_type):
	if model_type == 'vgg16':
		p = decode_predictions(pred) 
		print(p[0][0]) 
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
	return x 


def test_image(model, model_type, image_name):
	preprocess = alexnet_preprocess if model_type == 'alexnet' else vgg16_preprocess
	input_shape = none_tuple_replace(model.layers[0].input_shape)
	im = Image.open(image_name)  
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
		print('  Models: -alexnet -vgg16')
		print('  Options: -train') 
		sys.exit(1) 
	
	name = sys.argv[1] 
	if name[0] == '-': name = name[1:]
	if name != 'alexnet' and name != 'vgg16':
		print('Model type', sys.argv[1], 'does not match known model.')
		print('Known models: -alexnet -vgg16') 
		sys.exit(1) 
	
	train = (sys.argv[2] == '-train') if len(sys.argv) == 3 else False
	model = create_alexnet(train) if name == 'alexnet' else create_vgg16(train) 
	
	if not train: 
		test_files = ['dog.jpg'] if name == 'vgg16' else ['mini_dog.jpg', 'mini_horse.jpg', 'mini_car.jpg'] 
		for test_file in test_files: 
			test_image(model, name, 'data/' + test_file) 