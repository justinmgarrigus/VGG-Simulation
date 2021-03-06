#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include "ndarray.h"

void matrix_multiply(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D);

// Random value between [min, max] 
float rand_float(int min, int max) {
	return (float)rand() / (float)(RAND_MAX / (max - min)) + min;
}

int main_old() {
	const int input_size = 4000; 
	const int output_size = 1000;
	printf("input_size: %d, output_size: %d\n", input_size, output_size); 

	int input_shape[2] = { 1, input_size };
	ndarray* input = ndarray_create(2, input_shape); 
	for (int i = 0; i < input_size; i++) input->arr[i] = rand_float(-1, 1);

	int weight_shape[2] = { input_size, output_size };
	ndarray* weights = ndarray_create(2, weight_shape); 
	for (int i = 0; i < input_size * output_size; i++) weights->arr[i] = rand_float(-1, 1);

	int bias_shape[2] = { 1, output_size }; 
	ndarray* biases = ndarray_create(2, bias_shape); 
	for (int i = 0; i < output_size; i++) biases->arr[i] = rand_float(-1, 1); 

	int result_shape[2] = { 1, output_size }; 
	ndarray* result = ndarray_create(2, result_shape); 

	matrix_multiply(input, weights, biases, result);

	float highest_error = 0; 
	for (int i = 0; i < weight_shape[1]; i++) {
		float temp = biases->arr[i];
		for (int j = 0; j < weight_shape[0]; j++) {
			temp += input->arr[j] * weights->arr[i + j * weight_shape[1]];
		}

		float error = fabs(temp - result->arr[i]); 
		if (error > highest_error) highest_error = error; 
	}
	printf("Highest error: %f\n", highest_error); 

	ndarray_free(input); 
	ndarray_free(weights);
	ndarray_free(biases);
	ndarray_free(result);
}