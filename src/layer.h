#ifndef LAYER_H
#define LAYER_H 

#include "ndarray.h" 

typedef struct layer layer; 

enum layer_type {
	layer_type_convolutional = 1, 
	layer_type_max_pooling = 2, 
	layer_type_flatten = 3, 
	layer_type_dense = 4
}; 

enum layer_activation {
	layer_activation_relu = 1, 
	layer_activation_softmax = 2
};

typedef struct layer {
	int weight_set_count;
	ndarray **weights;
	void (*feed)(layer* input, layer* op);
	float (*activation)(float value);
	ndarray *outputs;
} layer; 

layer* layer_create(int weight_set_count, int* weight_set_shape_count, int** weight_set_shapes, enum layer_type type, enum layer_activation activation); 
void layer_free(layer* layer); 

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer);
void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer); 
void layer_flatten_feedforward(layer* input_layer, layer* flatten_layer); 
void layer_dense_feedforward(layer* input_layer, layer* dense_layer); 

float layer_relu(float value); 
float layer_softmax(float value); 

#endif 