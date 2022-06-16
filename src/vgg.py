# Disables initial "cuda device not found" warning message 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from audioop import bias
from cmath import inf
from unittest import result
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image 
import numpy as np
import sys
import tensorflow as tf 
import random
import struct 

import time
def current_milli_time():
    return round(time.time() * 1000)


model = VGG16(weights='imagenet') 
im = Image.open('/Users/bora/Desktop/dog.jpg') 
size = im.size

print(model.summary()) 

# Preprocessing: getting the average R, G, and B values
avg = [0, 0, 0]  
for x in range(size[0]): 
    for y in range(size[1]): 
        pixel = im.getpixel((x, y)) 
        for p in range(3): 
            avg[p] += pixel[p] 
for p in range(3): 
    avg[p] /= size[0] * size[1] 
    avg[p] = int(avg[p])

# Condensing image into a BGR 224x224 grid, subtracting averages from colors
image_input = np.zeros(shape=(1, 224, 224, 3)) 
for x in range(224):
    for y in range(224): 
        pixel = im.getpixel((x / 224 * size[0], y / 224 * size[1])) 
        for p in range(3): 
           image_input[0][y][x][p] = pixel[p] - avg[p]
        
        # Flipping R and B
        temp = image_input[0][y][x][2]
        image_input[0][y][x][2] = image_input[0][y][x][0] 
        image_input[0][y][x][0] = temp 


def relu(X):
   return np.maximum(0,X)


def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum


def shape_fix(shape): 
    l = list(shape) 
    for item in range(len(shape)): 
        if shape[item] == None: l[item] = 1 
        else: l[item] = shape[item]
    return tuple(l)


def conv_2D(layer, inputs):
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
    

def max_pooling_2D(layer, inputs):
    outputs = layer(inputs)
    inputsnp = inputs.numpy()
    outputsnp = outputs.numpy()
    
    for offset_x in range (0, inputsnp.shape[1], 2):
        for offset_y in range (0, inputsnp.shape[1], 2):
            for z in range (0, inputsnp.shape[3]):
                max_value = float('-inf')
                for kernel_x in range (2):
                    for kernel_y in range (2):
                        if inputsnp[0][offset_x + kernel_x][offset_y + kernel_y][z] > max_value:
                            max_value = inputsnp[0][offset_x + kernel_x][offset_y + kernel_y][z]
                outputsnp[0][offset_x//2][offset_y//2][z] = max_value


    
    eager_tensor = tf.convert_to_tensor(outputsnp, dtype=np.float32)
    return eager_tensor
    
    
        
def flatten(layer, inputs): 
    result_array = np.empty(shape=shape_fix(layer.output_shape)) 
    i = 0 
    for x in np.nditer(inputs.numpy()): 
        result_array[0][i] = x
        i += 1 
    eager_tensor = tf.convert_to_tensor(result_array, dtype=np.float32)
    return eager_tensor 
    
    
def dense(layer, inputs):
    outputs = layer(inputs)
    result_array = np.zeros(shape=shape_fix(layer.output_shape))
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


def save_network(model):
    file = open("network.nn", "wb")
    magic_number = 1234
    file.write(magic_number.to_bytes(4, byteorder='big', signed=True))
    number_of_layers = len(model.layers)
    file.write(number_of_layers.to_bytes(4, byteorder='big', signed=True))
    for layer in model.layers:
        if ('input' in str(layer.name)):
            layer_type = 0
            file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
        elif ('conv' in str(layer.name)):
            layer_type = 1
            file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
            if ('relu' in str(layer.activation)):
                type_of_activation = 1
                file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
            elif ('softmax' in str(layer.activation)):
                type_of_activation = 2
                file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
        elif ('pool' in str(layer.name)):
            layer_type = 2
            file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
        elif ('flatten' in str(layer.name)):
            layer_type = 3
            file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
        elif ('fc' in str(layer.name) or 'predictions' in str(layer.name)):
            layer_type = 4
            file.write(layer_type.to_bytes(4, byteorder='big', signed=True))
            if ('relu' in str(layer.activation)):
                type_of_activation = 1
                file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
            elif ('softmax' in str(layer.activation)):
                type_of_activation = 2
                file.write(type_of_activation.to_bytes(4, byteorder='big', signed=True))
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
    print("DONE!!!")
    file.close()


def load_network(model):
    file = open("network.nn", "rb")
    magic_number = int.from_bytes(file.read(4), byteorder='big', signed=True)
    number_of_layers = int.from_bytes(file.read(4), byteorder='big', signed=True)
    for layer_index in range(number_of_layers):
        layer_type = int.from_bytes(file.read(4), byteorder='big', signed=True)
        if (layer_type == 1):
            type_of_activation = int.from_bytes(file.read(4), byteorder='big', signed=True)
        elif (layer_type == 4):
            type_of_activation = int.from_bytes(file.read(4), byteorder='big', signed=True)
        number_of_weight_indices = int.from_bytes(file.read(4), byteorder='big', signed=True)
        for weight_index in range(number_of_weight_indices):
            length_of_weights_shape = int.from_bytes(file.read(4), byteorder='big', signed=True)
            weights_shape = []
            for shape_index in range(length_of_weights_shape):
                shape_value = int.from_bytes(file.read(4), byteorder='big', signed=True)
                weights_shape.append(shape_value) 
            number_of_data_stored_in_shape = 1
            for i in range(len(weights_shape)):
                number_of_data_stored_in_shape *= weights_shape[i]
            for i in range(number_of_data_stored_in_shape):
                ba = file.read(4)
                ba = struct.unpack("f", ba)[0]
    file.close()


if __name__ == '__main__': 
    # Feedforward
    x = model.layers[0](image_input) # Setting inputs 
    for layer in model.layers[1:]:   # Feed forward each layer 
        name = layer.__class__.__name__
        if name == 'Conv2D':         x = conv_2D(layer, x)
        elif name == 'MaxPooling2D': x = max_pooling_2D(layer, x) 
        elif name == 'Flatten':      x = flatten(layer, x) 
        elif name == 'Dense':        x = dense(layer, x)
        else: 
            print('Unrecognized layer:', name)
            sys.exit(0) 
      
    # Displays output layer
    print('Predictions:') 
    pred = decode_predictions(x.numpy()) 
    for i in range(5):
        print(' ', pred[0][i]) 