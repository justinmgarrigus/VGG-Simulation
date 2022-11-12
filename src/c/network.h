#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h" 

enum model_type { model_alexnet, model_vgg16 }; 

typedef struct network {
	int layer_count; 
	layer **layers; 
	char **labels; 
} network; 

network* network_create(enum model_type type); 
void network_free(network* network); 
void network_feedforward(network* network, ndarray* inputs); 
void network_decode_output(network* network); 

extern char *model_name; // Static (assumes only one network). TODO: think of a better solution than this. 

#if defined(SAVE_INTERMEDIATE) 
extern int conv_layer_counter; 
#endif 

#endif 