print("Started") 

from keras.applications.vgg16 import VGG16 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image 
import numpy as np
import sys

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

# Feedforward
x = model.layers[0](image_input) # Setting inputs 
for layer in model.layers[1:]:   # Feed forward each layer 
    x = layer(x)
  
# Displays output layer 
print(decode_predictions(x.numpy()))