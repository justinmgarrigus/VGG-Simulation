#include <stdio.h> 
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "cudaTensorCoreGemm.cuh" 

extern "C" {
	#include "layer_gpu.cuh" 
}

// TODO move this to ndarray.c
__device__ int ndarray_index(ndarray* nd, int* pos) {
	int index = 0;
	for (int i = 0; i < nd->dim; i++)
		index += nd->cumulative[i] * pos[i];
	return index;
}

__host__ void layer_convolutional_feedforward_gpu_setup(layer* input_layer, layer* conv_layer) {
	// Max thread count 
	cudaDeviceProp prop; 
	cudaGetDeviceProperties(&prop, 0); 
	int max_threads_per_block = prop.maxThreadsPerBlock; 
	
	// Move outputs to the host
	ndarray *h_outputs = ndarray_copy(conv_layer->outputs, cudaMemcpyDeviceToHost);
	
	int x_dim = h_outputs->shape[1];
	int y_dim = h_outputs->shape[2];
	int filters = h_outputs->shape[3];

	int things_to_process = x_dim * y_dim * filters;
	int threads = things_to_process > max_threads_per_block ? max_threads_per_block : things_to_process; 
	int blocks = things_to_process / threads + 1; 
	
	// Move inputs to the host
	ndarray *h_inputs = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);

	// Pad input array
	int padding[4] = {0, 1, 1, 0};
	ndarray *h_inputs_padded = ndarray_pad(h_inputs, padding, threads * blocks - things_to_process);

	// Move back to the device
	ndarray *d_inputs = ndarray_copy(h_inputs_padded, cudaMemcpyHostToDevice);
	
	printf("--Conv2D (%d, %d)\n", blocks, threads); 
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

__global__ void layer_convolutional_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weights, int blocks, int threads) {
	// TODO: assuming it will always be 1024. 
	ndarray* kernel = weights[0];
	ndarray* bias = weights[1];

	int x_dim = outputs->shape[1];
	int y_dim = outputs->shape[2];
	int index = blockIdx.x * threads + threadIdx.x;
	int filter = min(index / (x_dim * y_dim), kernel->shape[3] - 1);
	int dim = index % (x_dim * y_dim);
	int x = dim % x_dim;
	int y = dim / x_dim;

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

__host__ void layer_dense_feedforward_gpu_setup(layer* input_layer, layer* dense_layer) {
	printf("--Dense\n"); 
	
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
	
	matrix_multiply(h_inputs, h_outputs, NULL, NULL); 

	layer_dense_feedforward_gpu<<<blocks, threads>>>(
		input_layer->outputs, dense_layer->outputs, dense_layer->weights, blocks, threads);
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
		exit(1);
	}
	cudaDeviceSynchronize();
	
	switch (dense_layer->activation) {
		case layer_activation_relu: 
			layer_dense_gpu_relu<<<1,1>>>(dense_layer->outputs); 
			break; 
		
		case layer_activation_softmax: 
			layer_dense_gpu_divsum<<<1,1>>>(dense_layer->outputs); 
			break; 
			
		default: 
			fprintf(stderr, "Error: unrecognized activation function: %d\n", dense_layer->activation); 
			exit(1); 
	}
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
		exit(1);
	}
	cudaDeviceSynchronize();

	ndarray_free(h_outputs);
	ndarray_free(h_inputs);
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

__device__ ND_TYPE layer_activation(ND_TYPE* value, enum layer_activation activation) {
	switch (activation) {
		case layer_activation_relu:
			return layer_relu(value); 
		case layer_activation_softmax: 
			return layer_softmax(value); 
		default: 
			return 0; 
	}
} 

__device__ ND_TYPE layer_relu(ND_TYPE* value) { 
	return fmaxf(*value, 0);
}

__device__ ND_TYPE layer_softmax(ND_TYPE* value) {
	return expf(value[0]) / value[1];
}