#ifndef LAYER_GPU 
#define LAYER_GPU 

#include "ndarray.h"
#include "layer.h"

__host__ void layer_convolutional_feedforward_gpu_setup(layer* input_layer, layer* conv_layer);
__device__ int ndarray_index(ndarray* nd, int* pos);
__global__ void layer_convolutional_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weights, int blocks, int threads);
__host__ void layer_dense_feedforward_gpu_setup(layer* input_layer, layer* dense_layer);  
__global__ void layer_dense_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weight_set, int blocks, int threads); 
void layer_dense_gpu_divsum(ndarray* dense); 
void layer_dense_gpu_relu(ndarray* dense);
__global__ void layer_dense_gpu_divsum_device(ndarray* dense); 
__global__ void layer_dense_gpu_relu_device(ndarray* dense); 

#endif 