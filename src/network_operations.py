# Disables initial "cuda device not found" warning message 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.applications.vgg16 import VGG16 
import struct 
import sys 


def save_network(model, network_file_name):
	file = open(network_file_name, "wb")
	magic_number = 1234
	file.write(magic_number.to_bytes(4, byteorder='big', signed=True))
	number_of_layers = len(model.layers)
	file.write(number_of_layers.to_bytes(4, byteorder='big', signed=True))
	for layer in model.layers:
		print("Saving layer:", str(layer.name))
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
	file.close()
	print("Network saved to", network_file_name) 
	
	
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
	

def print_options(): 
	print("Options:")
	print("  -save")

	
if __name__ == '__main__': 
	if len(sys.argv) < 2: 
		print_options() 
	else: 
		if sys.argv[1] == '-save': 
			network_file_name = 'data/network.nn' if len(sys.argv) <= 2 else sys.argv[2] 
			if not network_file_name.startswith('data/'): network_file_name = 'data/' + network_file_name
			save_network(VGG16(weights='imagenet'), network_file_name) 
		else:
			print("Unknown option:", sys.argv[1])
			print_options() 