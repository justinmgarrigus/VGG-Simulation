#include <stdlib.h> 
#include <stdio.h> 
#include "network.h"
#include "layer.h" 

int random_float_01() {
	return (float)rand() / (float)RAND_MAX;
}

network* network_create(char* data_file, char* label_file) {
	network* network = malloc(sizeof(network)); 
	
	// For now, data_file is ignored. TODO
	network->layer_count = 1;
	network->layers = malloc(sizeof(layer*) * network->layer_count);
	
	// First layer from example ("Red")
	int red_weight_set_count = 1; 
	
	int *red_weight_set_shape_count = malloc(sizeof(int));
	red_weight_set_shape_count[0] = 3;
	
	int **red_weight_set_shapes = malloc(sizeof(int*));
	red_weight_set_shapes[0] = malloc(sizeof(int) * 3);
	red_weight_set_shapes[0][0] = 2; 
	red_weight_set_shapes[0][1] = 1; 
	red_weight_set_shapes[0][2] = 2; 
	
	enum layer_type red_type = layer_type_dense; 
	enum layer_activation red_activation = layer_activation_relu; 
	network->layers[0] = layer_create(red_weight_set_count, red_weight_set_shape_count, red_weight_set_shapes, red_type, red_activation); 
	free(red_weight_set_shape_count); 
	free(red_weight_set_shapes[0]);
	free(red_weight_set_shapes);

	// Second layer from example ("Blue")
	int blue_weight_set_count = 2; 
	
	int *blue_weight_set_shape_count = malloc(sizeof(int) * 2);
	blue_weight_set_shape_count[0] = 2;
	blue_weight_set_shape_count[1] = 3; 
	
	int **blue_weight_set_shapes = malloc(sizeof(int*) * 2);
	blue_weight_set_shapes[0] = malloc(sizeof(int) * 2);
	blue_weight_set_shapes[0][0] = 2; 
	blue_weight_set_shapes[0][1] = 3;
	blue_weight_set_shapes[1] = malloc(sizeof(int) * 3); 
	blue_weight_set_shapes[1][0] = 3; 
	blue_weight_set_shapes[1][1] = 2;
	blue_weight_set_shapes[1][2] = 1; 
	
	enum layer_type blue_type = layer_type_dense; 
	enum layer_activation blue_activation = layer_activation_relu; 
	network->layers[1] = layer_create(blue_weight_set_count, blue_weight_set_shape_count, blue_weight_set_shapes, blue_type, blue_activation); 
	free(blue_weight_set_shape_count);
	free(blue_weight_set_shapes[0]);
	free(blue_weight_set_shapes[1]);  
	free(blue_weight_set_shapes); 
	
	return network; 
}

void network_free(network* network) {
	for (int i = 0; i < network->layer_count; i++) 
		layer_free(network->layers[i]);
	free(network->layers);
	free(network);
}