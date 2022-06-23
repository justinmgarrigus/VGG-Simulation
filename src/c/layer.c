#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include "layer.h" 

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer) { printf("Conv\n"); }
void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer) { printf("Max\n"); }
void layer_flatten_feedforward(layer* input_layer, layer* flatten_layer) { printf("Flatten\n"); }
void layer_dense_feedforward(layer* input_layer, layer* dense_layer) { printf("Dense\n"); }

float layer_relu(float value) { return 0.0f; }
float layer_softmax(float value) { return 0.0f; }

layer* layer_create(int weight_set_count, ndarray** weights, enum layer_type type, enum layer_activation activation, ndarray* outputs) {
	layer *lr = malloc(sizeof(layer));
	
	lr->weight_set_count = weight_set_count; 
	lr->weights = weights; 
	lr->outputs = outputs; 
	
	switch (type) {
		case layer_type_none: 
			break; 
		
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
	
	switch (activation) {
		case layer_activation_none: 
			break; 
		
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
	
	return lr; 
}

void layer_free(layer* layer) {
	for (int i = 0; i < layer->weight_set_count; i++)
		ndarray_free(layer->weights[i]);
	free(layer->weights);
	ndarray_free(layer->outputs);
	free(layer);
}