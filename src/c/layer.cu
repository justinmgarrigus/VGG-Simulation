#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 

// Cuda uses a C++ compiler; this tells them to compile these files as if they 
// were using a C-style compiler.
extern "C" {
	#include "layer.cuh"
}

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
	ndarray **weights = (ndarray**)malloc(sizeof(ndarray*) * layer->weight_set_count); 
	cudaMemcpy(weights, layer->weights, sizeof(ndarray*) * layer->weight_set_count, cudaMemcpyDeviceToHost);
	for (int i = 0; i < layer->weight_set_count; i++)
		ndarray_free_gpu(weights[i]);
	free(weights); 
	cudaFree(layer->weights); 

	ndarray_free_gpu(layer->outputs);
	free(layer);
}

void layer_log(layer* layer, int index) {
	ndarray *outputs = ndarray_copy(layer->outputs, cudaMemcpyDeviceToHost);
	char buffer[20];
	snprintf(buffer, 20, "data/logs/c%d.txt", index);
	ndarray_log(outputs, buffer);
	ndarray_free(outputs);
}

// TODO move this to ndarray.c
__device__ int ndarray_index(ndarray* nd, int* pos) {
	int index = 0;
	for (int i = 0; i < nd->dim; i++)
		index += nd->cumulative[i] * pos[i];
	return index;
}

__global__ void layer_convolutional_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weights, int blocks, int threads) {
	// TODO: assuming it will always be 1024. 
	ndarray* kernel = weights[0];
	ndarray* bias = weights[1];

	int x_dim = outputs->shape[1];
	int y_dim = outputs->shape[2]; 
	int filter_count = outputs->shape[3];
	int index = blockIdx.x * threads + threadIdx.x;
	int filter = index / (x_dim * y_dim);
	int dim = index % (x_dim * y_dim);
	int x = dim % x_dim;
	int y = dim / x_dim;

	if (filter < kernel->shape[3]) {
		ND_TYPE result = bias->arr[filter];
		for (int kernel_x = 0; kernel_x < kernel->shape[0]; kernel_x++) {
			for (int kernel_y = 0; kernel_y < kernel->shape[1]; kernel_y++) {
				for (int channel = 0; channel < inputs->shape[3]; channel++) {
					int kernel_index[4] = { kernel_x, kernel_y, channel, filter };
					int inputs_index[4] = { 0, x + kernel_x, y + kernel_y, channel };
					result +=
						kernel->arr[ndarray_index(kernel, kernel_index)] *
						inputs->arr[ndarray_index(inputs, inputs_index)];
				}
			}
		}
		result = fmaxf(0, result); // TODO should be activation function instead

		int outputs_index[4] = { 0, x, y, filter };
		outputs->arr[ndarray_index(outputs, outputs_index)] = result;
	}
}

void layer_convolutional_feedforward(layer* input_layer, layer* conv_layer) {
	printf("Conv2D\n"); 

	// Max thread count 
	cudaDeviceProp prop; 
	cudaGetDeviceProperties(&prop, 0); 
	int max_threads_per_block = prop.maxThreadsPerBlock; 

	// Move outputs to the host
	ndarray *h_outputs = ndarray_copy(conv_layer->outputs, cudaMemcpyDeviceToHost);
	
	// Move inputs to the host
	ndarray *h_inputs = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);

	// Pad input array
	int padding[4] = {0, 1, 1, 0};
	ndarray *h_inputs_padded = ndarray_pad(h_inputs, padding);

	// Move back to the device
	ndarray *d_inputs = ndarray_copy(h_inputs_padded, cudaMemcpyHostToDevice);

	int x_dim = h_outputs->shape[1];
	int y_dim = h_outputs->shape[2];
	int filters = h_outputs->shape[3];

	int things_to_process = x_dim * y_dim * filters; 
	int threads = things_to_process > max_threads_per_block ? max_threads_per_block : things_to_process; 
	int blocks = things_to_process / threads + 1; 

	layer_convolutional_feedforward_gpu<<<blocks, threads>>>(
		d_inputs, conv_layer->outputs, conv_layer->weights, blocks, threads); 
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError())); 
		exit(1); 
	}
	cudaDeviceSynchronize();

	ndarray_free(h_outputs); 
	ndarray_free(h_inputs);
	ndarray_free(h_inputs_padded);
	ndarray_free_gpu(d_inputs); 
}

void layer_max_pooling_feedforward(layer* input_layer, layer* pool_layer) { 
	printf("MaxPooling\n");
	ndarray *input = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost); 
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
	printf("Flatten\n");
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

__global__ void layer_dense_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weight_set, int blocks, int threads) {
	// TODO: assuming it will always be 1024. 
	ndarray *weights = weight_set[0];
	ndarray *biases = weight_set[1];

	int j = blockIdx.x * threads + threadIdx.x;
	if (j < weights->shape[1]) {
		ND_TYPE result = biases->arr[j];
		for (int k = 0; k < weights->shape[0]; k++) {
			int input_index[2] = { 0, k }; 
			int weights_index[2] = { k, j };
			result +=
				inputs->arr[ndarray_index(inputs, input_index)] * 
				weights->arr[ndarray_index(weights, weights_index)];
		}
		int output_index[2] = { 0, j }; 
		outputs->arr[ndarray_index(outputs, output_index)] = result;
	}
}

__global__ void layer_dense_gpu_divsum(ndarray* dense) {
	ND_TYPE sum = 0; 
	for (int i = 0; i < dense->shape[1]; i++)
		sum += expf(dense->arr[i]);

	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = expf(dense->arr[i]) / sum; 
}

__global__ void layer_dense_gpu_relu(ndarray* dense) {
	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = fmaxf(dense->arr[i], 0); 
}

int dense_counter = 0; // TODO remove, replace with function pointer. 
void layer_dense_feedforward(layer* input_layer, layer* dense_layer) { 
	printf("Dense\n"); 
	
	// Max thread count 
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int max_threads_per_block = prop.maxThreadsPerBlock;

	// Move outputs to the host
	ndarray *h_outputs = ndarray_copy(dense_layer->outputs, cudaMemcpyDeviceToHost);

	// Move inputs to the host
	ndarray *h_inputs = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);

	int things_to_process = h_outputs->shape[1];
	int threads = things_to_process > max_threads_per_block ? max_threads_per_block : things_to_process;
	int blocks = things_to_process / threads + 1;

	layer_dense_feedforward_gpu<<<blocks, threads>>>(
		input_layer->outputs, dense_layer->outputs, dense_layer->weights, blocks, threads);
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
		exit(1);
	}
	cudaDeviceSynchronize();

	dense_counter++; 
	if (dense_counter < 3)
		layer_dense_gpu_relu<<<1,1>>>(dense_layer->outputs); 
	else 
		layer_dense_gpu_divsum<<<1,1>>>(dense_layer->outputs); 
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
		exit(1);
	}
	cudaDeviceSynchronize();

	ndarray_free(h_outputs);
	ndarray_free(h_inputs);
}

__device__ ND_TYPE layer_relu(ND_TYPE* value) { 
	return fmaxf(*value, 0);
}

__device__ ND_TYPE layer_softmax(ND_TYPE* value) {
	return expf(value[0]) / value[1];
}