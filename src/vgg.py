print("Started") 

from keras.applications.vgg16 import VGG16 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image 
import numpy as np
import sys
import tensorflow as tf 

model = VGG16(weights='imagenet') 
im = Image.open('data/dog.jpg') 
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


def shape_fix(shape): 
    l = list(shape) 
    for item in range(len(shape)): 
        if shape[item] == None: l[item] = 1 
        else: l[item] = shape[item]
    return tuple(l) 


def conv_2D(layer, inputs):
    outputs = layer(inputs) 
    return outputs
    

def max_pooling_2D(layer, inputs): 
    outputs = layer(inputs) 
    return outputs 
    
    
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
#    new_array = np.empty(shape=shape_fix(layer.output_shape)) 
#    eager_tensor = tf.convert_to_tensor(new_array, dtype=np.float32) 
    return outputs 


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
    print(decode_predictions(x.numpy()))