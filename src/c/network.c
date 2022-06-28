#include <stdlib.h> 
#include <stdio.h> 
#include <string.h>
#include <sys/stat.h>  
#include <stdint.h> 
#include "network.h"
#include "ndarray.h"
#include "json.h"

uint32_t file_read_int(FILE* file, unsigned char* buffer) {
	// Data is encoded in .nn files as 32-bit integers, but they can be saved as 16-bit
	// ints internally inside the network struct if the system architecture wants. 
	fread(buffer, sizeof(uint32_t), 1, file);
	return buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3]; 
}

float file_read_float(FILE* file, unsigned char* buffer) {
	fread(buffer, sizeof(float), 1, file); 
	float result; 
	memcpy(&result, buffer, sizeof(float));
	return result; 
}

char **get_labels_json(json_value* json) {
	int length = json->u.object.length;
	char **labels = malloc(sizeof(char*) * length); 
	for (int index = 0; index < length; index++) {
		// TODO this function is unsafe; check the data before retrieving it to ensure 
		// it is in the correct format! 
		json_object_entry *entry = &(json->u.object.values[index]); 
		char *str = entry->value->u.array.values[1]->u.string.ptr; 
		char *label = malloc(strlen(str) + 1); 
		strcpy(label, str); 
		labels[index] = label;
	}
	return labels; 
}

network* network_create(char* data_file, char* label_file) {
	FILE *file = fopen(data_file, "rb");
	if (file == NULL) {
		fprintf(stderr, "Error opening file %s\n", data_file); 
		exit(1); 
	}
	
	unsigned char *buffer = malloc(sizeof(int) > sizeof(float) ? sizeof(int) : sizeof(float)); 
	
	int magic_number = file_read_int(file, buffer); 
	if (magic_number != 1234) {
		fprintf(stderr, "Magic number is %d, 1234 expected. Check if file is corrupted.\n", magic_number); 
		exit(1); 
	}
	
	network* net = malloc(sizeof(net));
	net->layer_count = file_read_int(file, buffer);
	net->layers = malloc(sizeof(layer*) * net->layer_count);
	
	for (int i = 0; i < net->layer_count; i++) {
		int layer_type = file_read_int(file, buffer); 
		int activation_type; 
		if (layer_type == layer_type_convolutional || layer_type == layer_type_dense)
			activation_type = file_read_int(file, buffer);
		else 
			activation_type = 0; 
		
		int weight_set_count = file_read_int(file, buffer);  
		ndarray **weight_set = malloc(sizeof(ndarray*) * weight_set_count); 
		for (int set_index = 0; set_index < weight_set_count; set_index++) {
			int dimensions = file_read_int(file, buffer); 
			int *shape = malloc(sizeof(int) * dimensions); 
			for (int dimension = 0; dimension < dimensions; dimension++)
				shape[dimension] = file_read_int(file, buffer); 
			
			int *counter = malloc(sizeof(int) * dimensions); 
			for (int c = 0; c < dimensions; c++) 
				counter[c] = 0; 
			
			ndarray *weights = ndarray_create(dimensions, shape);
			do {
				ndarray_set_val_list(weights, counter, file_read_float(file, buffer)); 
			}
			while (ndarray_decimal_count(dimensions, counter, shape)); 
			free(counter); 
			
			weight_set[set_index] = weights; 
		}
		
		int output_length = file_read_int(file, buffer); 
		int *output_shape = malloc(sizeof(int) * output_length);
		for (int output_index = 0; output_index < output_length; output_index++)
			output_shape[output_index] = file_read_int(file, buffer);
		ndarray *outputs = ndarray_create(output_length, output_shape); 
		
		net->layers[i] = layer_create(weight_set_count, weight_set, layer_type, activation_type, outputs); 
	}
	
	// Read labels
	struct stat file_status; 
	if (stat(label_file, &file_status) != 0) {
		fprintf(stderr, "File %s is not found\n", label_file); 
		exit(1); 
	}
	
	int file_size = file_status.st_size; 
	char *file_contents = malloc(sizeof(char) * file_size); 
	
	FILE *label_fp = fopen(label_file, "rt"); 
	if (label_fp == NULL) {
		fprintf(stderr, "Unable to open %s\n", label_file); 
		exit(1); 
	}
	
	if (fread(file_contents, file_size, 1, label_fp) != 1) {
		fprintf(stderr, "Unable to read contents of %s\n", label_file); 
		fclose(label_fp); 
		free(file_contents); 
		exit(1); 
	}
	fclose(label_fp); 
	
	json_value *json = json_parse((json_char*)file_contents, file_size); 
	if (json == NULL) {
		fprintf(stderr, "Unable to parse data\n");
		free(file_contents); 
		exit(1);
	}
	
	net->labels = get_labels_json(json); 
	
	json_value_free(json);
	free(file_contents); 
	
	return net; 
}

void network_feedforward(network* network, ndarray* inputs) {
#ifdef DRAW_PROGRESS
	network_operation_time = 0; 
#endif 
	
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
	for (int i = 1; i < network->layer_count; i++) {
		network->layers[i]->feed(network->layers[i-1], network->layers[i]); 
	}
}

void network_decode_output(network* network) {
	const int NUM_SCORES = 5; 
	ND_TYPE scores[6] = { 0 }; // length = NUM_SCORES + 1 
	char *labels[6]; 
 	
	ndarray *output = network->layers[network->layer_count-1]->outputs; 
	for (int label_index = 0; label_index < 1000 /* TODO */; label_index++) {
		ND_TYPE value = ndarray_get_val_param(output, 0, label_index); 
		scores[0] = value; 
		labels[0] = network->labels[label_index];
		for (int score_index = 1; score_index <= NUM_SCORES; score_index++) {
			if (scores[score_index] < scores[score_index-1]) {
				ND_TYPE temp_score = scores[score_index]; 
				scores[score_index] = scores[score_index-1]; 
				scores[score_index-1] = temp_score; 
				
				char *temp_label = labels[score_index]; 
				labels[score_index] = labels[score_index-1]; 
				labels[score_index-1] = temp_label;
			}
			else break; 
		}
	}
	
	for (int i = NUM_SCORES; i > 0; i--) {
		printf("%.5f, %s\n", scores[i], labels[i]); 
	}
}

void network_free(network* network) {
	for (int i = 0; i < network->layer_count; i++) 
		layer_free(network->layers[i]);
	free(network->layers);
	
	for (int i = 0; i < 1000 /* TODO */; i++) 
		free(network->labels[i]); 
	free(network->labels); 
	
	free(network);
}