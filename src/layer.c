#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include "layer.h" 

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer) { }
void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer) { }
void layer_flatten_feedforward(layer* input_layer, layer* flatten_layer) { }
void layer_dense_feedforward(layer* input_layer, layer* dense_layer) { }

float layer_relu(float value) { return 0.0f; }
float layer_softmax(float value) { return 0.0f; }

layer* layer_create(int weight_set_count, int* weight_set_shape_count, int** weight_set_shapes, enum layer_type type, enum layer_activation activation) {
	layer *lr = malloc(sizeof(layer));
	
	lr->weight_set_count = weight_set_count; 
	lr->weights = malloc(sizeof(ndarray*) * weight_set_count);
	for (int i = 0; i < weight_set_count; i++) 
		lr->weights[i] = ndarray_create(weight_set_shape_count[i], weight_set_shapes[i]);
	
	switch (activation) {
		case layer_activation_relu: 
			lr->activation = layer_relu; 
			break; 
			
		case layer_activation_softmax: 
			lr->activation = layer_softmax; 
			break; 
			
		default: 
			fprintf(stderr, "Unknown activation type specified: %d\n", activation); 
			exit(-1); 
	}
	
	switch (type) {
		case layer_type_convolutional: 
			lr->feed = layer_convolutional_feedforward; 
			break; 
		
		case layer_type_max_pooling: 
			lr->feed = layer_max_pooling_feedforward; 
			break; 
			
		case layer_type_flatten: 
			lr->feed = layer_flatten_feedforward; 
			break; 
		
		case layer_type_dense: 
			lr->feed = layer_dense_feedforward; 
			break; 
			
		default: 
			fprintf(stderr, "Unknown layer type specified: %d\n", type); 
			exit(-1); 
	}
	
	return lr; 
}

void layer_free(layer* layer) {
	for (int i = 0; i < layer->weight_set_count; i++)
		ndarray_free(layer->weights[i]);
	free(layer->weights);
	//ndarray_free(layer->outputs); // TODO
	free(layer);
}