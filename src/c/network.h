#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h" 

typedef struct network {
	int layer_count; 
	layer **layers; 
	char **labels; 
} network; 

network* network_create(char* data_file, char* label_file); 
void network_free(network* network); 
void network_feedforward(network* network, ndarray* inputs); 
void network_decode_output(network* network); 

#endif 