#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include "network.h"
#include "layer.h" 
#include "ndarray.h" 

int random_float_01() {
	return (float)rand() / (float)RAND_MAX;
}

int file_read_int(FILE* file) {
	const int buffer_size = sizeof(int); 
	unsigned char buffer[buffer_size]; 
	fread(buffer, buffer_size, 1, file); 
	return buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3]; 
}

float file_read_float(FILE* file) {
	const int buffer_size = sizeof(float); 
	unsigned char buffer[buffer_size]; 
	fread(buffer, buffer_size, 1, file); 
	float result; 
	memcpy(&result, buffer, buffer_size);
	return result; 
}

network* network_create(char* data_file, char* label_file) {
	FILE *file = fopen(data_file, "rb");
	if (file == NULL) {
		fprintf(stderr, "Error opening file %s\n", data_file); 
		exit(1); 
	}
	
	int magic_number = file_read_int(file); 
	if (magic_number != 1234) {
		fprintf(stderr, "Magic number is %d, 1234 expected. Check if file is corrupted.\n", magic_number); 
		exit(1); 
	}
	
	network* net = malloc(sizeof(net));
	net->layer_count = file_read_int(file);
	net->layers = malloc(sizeof(layer*) * net->layer_count);
	
	for (int i = 0; i < net->layer_count; i++) {
		int layer_type = file_read_int(file); 
		int activation_type; 
		if (layer_type == layer_type_convolutional || layer_type == layer_type_dense)
			activation_type = file_read_int(file);
		else 
			activation_type = 0; 
		
		int weight_set_count = file_read_int(file);  
		ndarray **weight_set = malloc(sizeof(ndarray*) * weight_set_count); 
		for (int set_index = 0; set_index < weight_set_count; set_index++) {
			int dimensions = file_read_int(file); 
			int *shape = malloc(sizeof(int) * dimensions); 
			for (int dimension = 0; dimension < dimensions; dimension++)
				shape[dimension] = file_read_int(file); 
			
			int *counter = malloc(sizeof(int) * dimensions); 
			for (int c = 0; c < dimensions; c++) 
				counter[c] = 0; 
			
			ndarray *weights = ndarray_create(dimensions, shape);
			do {
				ndarray_set_val_list(weights, counter, file_read_float(file)); 
			}
			while (ndarray_decimal_count(dimensions, counter, shape)); 
			free(counter); 
			
			weight_set[set_index] = weights; 
		}
		
		int output_length = file_read_int(file); 
		int *output_shape = malloc(sizeof(int) * output_length);
		for (int output_index = 0; output_index < output_length; output_index++)
			output_shape[output_index] = file_read_int(file);
		ndarray *outputs = ndarray_create(output_length, output_shape); 
		
		net->layers[i] = layer_create(weight_set_count, weight_set, layer_type, activation_type, outputs); 
	}
	
	return net; 
}

void network_feedforward(network* network, ndarray* inputs) {
	// First layer is the input layer; copy inputs over. 
	int *counter = malloc(sizeof(int) * inputs->dim); 
	for (int c = 0; c < inputs->dim; c++) 
		counter[c] = 0; 
	do {
		ndarray_set_val_list(network->layers[0]->outputs, counter, ndarray_get_val_list(inputs, counter)); 
	}
	while (ndarray_decimal_count(inputs->dim, counter, inputs->shape)); 
	free(counter); 
	
	// Feed the values forward. 
	int file_counter = 0; 
	char buffer[100]; 
	for (int i = 1; i < network->layer_count; i++) {
		sprintf(buffer, "logs/c_log_%d.txt", file_counter++); 
		ndarray_log(network->layers[i-1]->outputs, buffer); 
		network->layers[i]->feed(network->layers[i-1], network->layers[i]); 
	}
	
	sprintf(buffer, "logs/c_log_%d.txt", file_counter); 
	ndarray_log(network->layers[network->layer_count-1]->outputs, buffer); 
}

void network_free(network* network) {
	for (int i = 0; i < network->layer_count; i++) 
		layer_free(network->layers[i]);
	free(network->layers);
	free(network);
}