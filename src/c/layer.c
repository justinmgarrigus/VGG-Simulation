#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "layer.h" 
#include "layer_gpu.cuh" 

layer* layer_create(int weight_set_count, ndarray** weights, enum layer_type type, enum layer_activation activation, ndarray* outputs) {
	layer *lr = (layer*)malloc(sizeof(layer));
	lr->weight_set_count = weight_set_count; 
	
	ndarray **copied_weights = (ndarray**)malloc(sizeof(ndarray*) * weight_set_count); 
	for (int i = 0; i < weight_set_count; i++)
		copied_weights[i] = ndarray_copy(weights[i], cudaMemcpyHostToDevice); 
	cudaMalloc(&(lr->weights), sizeof(ndarray*) * weight_set_count); 
	cudaMemcpy(lr->weights, copied_weights, sizeof(ndarray*) * weight_set_count, cudaMemcpyHostToDevice); 
	free(copied_weights); 
	
	lr->outputs = ndarray_copy(outputs, cudaMemcpyHostToDevice); 

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
			
		case layer_type_batch_normalization:
			lr->feed = layer_batch_normalization_feedforward; 
			break; 
			
		default: 
			fprintf(stderr, "Unknown layer type specified: %d\n", type); 
			exit(-1); 
	}
	
	lr->activation = activation; 
	
	return lr; 
}

void layer_free(layer* layer) {
	ndarray **weights = (ndarray**)malloc(sizeof(ndarray*) * layer->weight_set_count); 
	cudaMemcpy(weights, layer->weights, sizeof(ndarray*) * layer->weight_set_count, cudaMemcpyDeviceToHost);
	for (int i = 0; i < layer->weight_set_count; i++)
		ndarray_free_gpu(weights[i]);
	free(weights); 
	cudaFree(layer->weights); 

	ndarray_free_gpu(layer->outputs);
	free(layer);
}

ndarray** layer_copy_weights(layer* lr, enum cudaMemcpyKind kind) {
	ndarray **copied_weights;
	if (kind == cudaMemcpyDeviceToHost) {
		ndarray **host_weights = malloc(sizeof(ndarray*) * lr->weight_set_count); 
		cudaMemcpy(host_weights, lr->weights, sizeof(ndarray*) * lr->weight_set_count, kind); 
		copied_weights = malloc(sizeof(ndarray*) * lr->weight_set_count);
		for (int i = 0; i < lr->weight_set_count; i++)
			copied_weights[i] = ndarray_copy(host_weights[i], kind);
		free(host_weights); 
	}
	else if (kind == cudaMemcpyHostToDevice) {
		ndarray **host_weights = malloc(sizeof(ndarray*) * lr->weight_set_count); 
		for (int i = 0; i < lr->weight_set_count; i++) 
			host_weights[i] = ndarray_copy(lr->weights[i], kind); 
		cudaMalloc(&copied_weights, sizeof(ndarray*) * lr->weight_set_count);
		cudaMemcpy(copied_weights, host_weights, sizeof(ndarray*) * lr->weight_set_count, kind);
		free(host_weights);
	}
	return copied_weights; 
}

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer) {
	layer_convolutional_feedforward_gpu_setup(input_layer, conv_layer); 
}

void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer) { 
	printf("--MaxPooling\n");
	printf("input\n"); 
	ndarray *input = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost); 
	printf("output\n"); 
	ndarray *output = ndarray_copy(pool_layer->outputs, cudaMemcpyDeviceToHost);
	
	for (int offset_x = 0; offset_x < input->shape[1]; offset_x += 2) {
		for (int offset_y = 0; offset_y < input->shape[2]; offset_y += 2) {
			for (int z = 0; z < input->shape[3]; z++) {
				ND_TYPE max_value = 0; 
				for (int kernel_x = 0; kernel_x < 2; kernel_x++) {
					for (int kernel_y = 0; kernel_y < 2; kernel_y++) {
						ND_TYPE val = ndarray_get_val_param(input, 0, offset_x + kernel_x, offset_y + kernel_y, z); 
						if (val > max_value) 
							max_value = val; 
					}
				}
				ndarray_set_val_param(output, max_value, 0, offset_x / 2, offset_y / 2, z); 
			}
		}
	}

	pool_layer->outputs = ndarray_copy(output, cudaMemcpyHostToDevice); 
}

void layer_flatten_feedforward(layer* input_layer, layer* flatten_layer) { 
	printf("--Flatten\n");
	ndarray *input = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost); 
	ndarray *output = ndarray_copy(flatten_layer->outputs, cudaMemcpyDeviceToHost); 
	
	int *pos = (int*)malloc(sizeof(int) * input->dim);
	for (int i = 0; i < input->dim; i++)
		pos[i] = 0;
	
	int index = 0; 
	do {
		ndarray_set_val_param(output, ndarray_get_val_list(input, pos), 0, index++);
	}
	while (ndarray_decimal_count(input->dim, pos, input->shape));
	free(pos); 

	flatten_layer->outputs = ndarray_copy(output, cudaMemcpyHostToDevice);
}

void layer_dense_feedforward(layer* input_layer, layer* dense_layer) { 
	layer_dense_feedforward_gpu_setup(input_layer, dense_layer); 
}

void layer_batch_normalization_feedforward(layer* input_layer, layer* batch_layer) {
	// TODO 
}