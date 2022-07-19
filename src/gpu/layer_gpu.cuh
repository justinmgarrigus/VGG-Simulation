#ifndef LAYER_GPU 
#define LAYER_GPU 

#include "ndarray.h"
#include "layer.h"

__host__ void layer_convolutional_feedforward_gpu_setup(layer* input_layer, layer* conv_layer);
__device__ int ndarray_index(ndarray* nd, int* pos);
__global__ void layer_convolutional_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weights, int blocks, int threads);
__host__ void layer_dense_feedforward_gpu_setup(layer* input_layer, layer* dense_layer);  
__global__ void layer_dense_feedforward_gpu(ndarray* inputs, ndarray* outputs, ndarray** weight_set, int blocks, int threads); 
__global__ void layer_dense_gpu_divsum(ndarray* dense); 
__global__ void layer_dense_gpu_relu(ndarray* dense); 
__device__ ND_TYPE layer_activation(ND_TYPE* value, enum layer_activation activation);
__device__ ND_TYPE layer_relu(ND_TYPE* value);
__device__ ND_TYPE layer_softmax(ND_TYPE* value);

#endif 