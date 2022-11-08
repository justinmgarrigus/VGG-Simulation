#include <stdio.h> 
#include <math.h> 
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "cudaTensorCoreGemm.cuh"
#include "io.h" 

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

__host__ void layer_convolutional_feedforward_gpu_setup_normal(layer* input_layer, layer* conv_layer) {
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

__host__ void layer_convolutional_feedforward_gpu_setup_tensorcore(layer* input_layer, layer* conv_layer) {
	ndarray* h_input = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);
	ndarray** h_weight_set = layer_copy_weights(conv_layer, cudaMemcpyDeviceToHost);
	ndarray* h_kernel = h_weight_set[0]; 
	ndarray* h_bias = h_weight_set[1]; 
	ndarray* h_padding = h_weight_set[2];
	
	printf("Input entropy: %f\n", ndarray_entropy(h_input));
	
	// Pad input array
	ndarray_deep_display(h_padding); 
 	if (h_padding->arr[1] != 0) {
		int padding[4] = {0, 1, 1, 0};
		ndarray *h_inputs_padded = ndarray_pad(h_input, padding, 1); 
		free(h_input); 
		h_input = h_inputs_padded; 
	}
	
	// Convert h_input to an image2col representation
	int kernel_length = h_kernel->shape[0]; 
	int stride_amount = h_weight_set[3]->arr[0];
	
	printf("Stride = %d\n", stride_amount);
	
	int width = floor(((h_input->shape[1] - kernel_length) / stride_amount) + 1); 
	int height = floor(((h_input->shape[2] - kernel_length) / stride_amount) + 1); 
	int cols_shape[2] = { width * height, h_input->shape[3] * kernel_length * kernel_length }; 
	ndarray *cols = ndarray_create(2, cols_shape);

	printf("Cols dimension: %d, %d\n", cols_shape[0], cols_shape[1]); 
	printf("h_kernel shape: [%d, %d, %d]\n", h_kernel->shape[2], h_kernel->shape[0], h_kernel->shape[1]); 
	
	int iter_shape[3] = { h_kernel->shape[2], h_kernel->shape[0], h_kernel->shape[1] }; 
	int pos[3]; 
	for (int vec_index = 0; vec_index < cols->shape[0]; vec_index++) {
		memset(pos, 0, sizeof(int) * 3); 
		int col_index = 0;
		int kernel_x = vec_index / height * stride_amount; 
		int kernel_y = vec_index % width  * stride_amount; 
		do { 
			float value = ndarray_get_val_param(h_input, 0, kernel_x + pos[1], kernel_y + pos[2], pos[0]); 
			ndarray_set_val_param(cols, value, vec_index, col_index++);  
		}
		while (ndarray_decimal_count(3, pos, iter_shape));
	}
	
	// Convert h_kernel into an image2col representation
	int kernel_col_shape[2] = { h_kernel->shape[0] * h_kernel->shape[1] * h_kernel->shape[2], h_kernel->shape[3] }; 
	ndarray *kernel_col = ndarray_create(2, kernel_col_shape);
	
	for (int c = 0; c < kernel_col->shape[1]; c++) {
		memset(pos, 0, sizeof(int) * 3); 
		int col_index = 0; 
		do {
			float value = ndarray_get_val_param(h_kernel, pos[1], pos[2], pos[0], c); 
			ndarray_set_val_param(kernel_col, value, col_index++, c);  
		}
		while (ndarray_decimal_count(3, pos, iter_shape));
	}
	
	// Create bias array (to simplify matrix multiplication)
	int bias_col_shape[2] = { cols->shape[0], kernel_col->shape[1] };
	ndarray *bias_col = ndarray_create(2, bias_col_shape); 
	for (int i = 0; i < bias_col->shape[0]; i++)
		for (int j = 0; j < bias_col->shape[1]; j++)
			ndarray_set_val_param(bias_col, h_bias->arr[j], i, j);
			
	// Multiply img2col input by img2col kernel 
	int output_col_shape[2] = { cols->shape[0], kernel_col->shape[1] }; 
	ndarray *output_col = ndarray_create(2, output_col_shape);
	matrix_multiply(cols, kernel_col, bias_col, output_col); 
	
	// Perform the activation function (relu) 
	for (int i = 0; i < output_col->count; i++) {
		if (output_col->arr[i] < 0) 
			output_col->arr[i] = 0; 
	}
	
	// Convert output (2D) back into a 4D matrix
	int length = (int)sqrt(output_col->shape[0]); 
	int output_shape[4] = { 1, length, length, output_col->shape[1] }; 
	ndarray *output = ndarray_create(4, output_shape); 
	
	for (int filter_index = 0; filter_index < output_col->shape[1]; filter_index++) {
		for (int i = 0; i < output_col->shape[0]; i++) { 
			float value = ndarray_get_val_param(output_col, i, filter_index); 
			ndarray_set_val_param(output, value, 0, i / length, i % length, filter_index); 
		}
	}
	
	conv_layer->outputs = ndarray_copy(output, cudaMemcpyHostToDevice);
	
	printf("Entropy:\n"); 
	printf("  cols: %f\n", ndarray_entropy(cols)); 
	printf("  kernel_col: %f\n", ndarray_entropy(kernel_col)); 
	printf("  output_col: %f\n", ndarray_entropy(output_col)); 
	printf("  output: %f\n", ndarray_entropy(output)); 
	
	ndarray_free(h_input); 
	free(h_weight_set); 
	ndarray_free(h_kernel); 
	ndarray_free(h_bias);
	ndarray_free(cols); 
	ndarray_free(kernel_col); 
	ndarray_free(bias_col); 
	ndarray_free(output_col);
	ndarray_free(output);
}

__host__ void layer_convolutional_feedforward_gpu_setup(layer* input_layer, layer* conv_layer) {
	printf("--Conv2D\n"); 
//	layer_convolutional_feedforward_gpu_setup_normal(input_layer, conv_layer); 
#ifdef TENSORCORE
	layer_convolutional_feedforward_gpu_setup_tensorcore(input_layer, conv_layer); 
#else 
	layer_convolutional_feedforward_gpu_setup_normal(input_layer, conv_layer); 
#endif 
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

void layer_dense_feedforward_gpu_setup_normal(layer* input_layer, layer* dense_layer) {
	// Max thread count 
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int max_threads_per_block = prop.maxThreadsPerBlock;

	// Move outputs to the host
	ndarray* h_outputs = ndarray_copy(dense_layer->outputs, cudaMemcpyDeviceToHost);

	// Move inputs to the host
	ndarray* h_inputs = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);

	int things_to_process = h_outputs->shape[1];
	int threads = things_to_process > max_threads_per_block ? max_threads_per_block : things_to_process;
	int blocks = things_to_process / threads + 1;

	layer_dense_feedforward_gpu << <blocks, threads >> > (
		input_layer->outputs, dense_layer->outputs, dense_layer->weights, blocks, threads);
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error in kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
		exit(1);
	}
	cudaDeviceSynchronize();

	switch (dense_layer->activation) {
		case layer_activation_relu:
			layer_dense_gpu_relu_device << <1, 1 >> > (dense_layer->outputs);
			break;

		case layer_activation_softmax:
			layer_dense_gpu_divsum_device << <1, 1 >> > (dense_layer->outputs);
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

void layer_dense_feedforward_gpu_setup_tensorcore(layer* input_layer, layer* dense_layer) {
	ndarray* input = ndarray_copy(input_layer->outputs, cudaMemcpyDeviceToHost);
	ndarray** weight_set = layer_copy_weights(dense_layer, cudaMemcpyDeviceToHost);
	ndarray* weights = weight_set[0];
	ndarray* biases = weight_set[1];

	int shape[2] = { 1, weights->shape[1] };
	ndarray* output = ndarray_create(2, shape);

	for (int i = 0; i < output->count; i++)
		output->arr[i] = 0;

	matrix_multiply(input, weights, biases, output);

	switch (dense_layer->activation) {
	case layer_activation_relu:
		layer_dense_gpu_relu(output);
		break;

	case layer_activation_softmax:
		layer_dense_gpu_divsum(output);
		break;

	default:
		fprintf(stderr, "Error: unrecognized activation function: %d\n", dense_layer->activation);
		exit(1);
	}

	dense_layer->outputs = ndarray_copy(output, cudaMemcpyHostToDevice);
	ndarray_free(input);
	ndarray_free(weights);
	ndarray_free(biases);
	ndarray_free(output);
}

__host__ void layer_dense_feedforward_gpu_setup(layer* input_layer, layer* dense_layer) {
	printf("--Dense\n");
	
	layer_dense_feedforward_gpu_setup_normal(input_layer, dense_layer); 
//#ifdef TENSORCORE
//	layer_dense_feedforward_gpu_setup_tensorcore(input_layer, dense_layer); 
//#else 
//	layer_dense_feedforward_gpu_setup_normal(input_layer, dense_layer); 
//#endif 

	ndarray *outputs = ndarray_copy(dense_layer->outputs, cudaMemcpyDeviceToHost); 
	printf("  Entropy: %f\n", ndarray_entropy(outputs)); 
	free(outputs); 
}

__global__ void layer_dense_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weight_set, int blocks, int threads) {
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

// TODO: don't copy functions, find a better solution.
__global__ void layer_dense_gpu_divsum_device(ndarray* dense) {
	ND_TYPE sum = 0;
	for (int i = 0; i < dense->shape[1]; i++)
		sum += expf(dense->arr[i]);

	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = expf(dense->arr[i]) / sum;
}

__global__ void layer_dense_gpu_relu_device(ndarray* dense) {
	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = fmaxf(dense->arr[i], 0);
}

void layer_dense_gpu_divsum(ndarray* dense) {
	ND_TYPE sum = 0; 
	for (int i = 0; i < dense->shape[1]; i++)
		sum += expf(dense->arr[i]);

	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = expf(dense->arr[i]) / sum; 
}

void layer_dense_gpu_relu(ndarray* dense) {
	for (int i = 0; i < dense->shape[1]; i++)
		dense->arr[i] = fmaxf(dense->arr[i], 0); 
}