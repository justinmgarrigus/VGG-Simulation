#include <stdio.h>
#include <stdlib.h> 
#include <stdarg.h> 
#include <string.h> 
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "ndarray.h" 

ndarray* ndarray_create(int dim, int* shape) {
	int *cumulative = malloc(sizeof(int) * dim); 
	cumulative[dim-1] = 1; 
	for (int i = dim-2; i >= 0; i--) 
		cumulative[i] = cumulative[i+1] * shape[i+1]; 
	int count = cumulative[0] * shape[0]; 
	ND_TYPE *arr = malloc(sizeof(ND_TYPE) * count); 
	
	ndarray *nd = malloc(sizeof(ndarray)); 
	nd->dim = dim; 
	nd->shape = malloc(sizeof(int) * dim); 
	memcpy(nd->shape, shape, sizeof(int) * dim); 
	nd->cumulative = cumulative; 
	nd->arr = arr; 
	nd->count = count; 
	return nd; 
}

	int dim; 
	int *shape; 
	int count; 
	int *cumulative; 
	ND_TYPE *arr; 

ndarray* ndarray_create_gpu(int dim, int* shape) {
	int *cumulative = malloc(sizeof(int) * dim); 
	cumulative[dim-1] = 1; 
	for (int i = dim-2; i >= 0; i--) 
		cumulative[i] = cumulative[i+1] * shape[i+1]; 
	int *d_cumulative; cudaMalloc(&d_cumulative, sizeof(int) * dim); 
	cudaMemcpy(d_cumulative, cumulative, sizeof(int) * dim, cudaMemcpyHostToDevice); 
	free(cumulative); 
	
	int count = cumulative[0] * shape[0]; 
	ND_TYPE *d_arr; cudaMalloc(&d_arr, sizeof(ND_TYPE) * count);

	int *d_shape; cudaMalloc(&d_shape, sizeof(int) * dim); 
	cudaMemcpy(d_shape, shape, sizeof(int) * dim, cudaMemcpyHostToDevice); 

	ndarray *nd = malloc(sizeof(ndarray)); 
	nd->dim = dim; 
	nd->shape = d_shape; 
	nd->cumulative = d_cumulative; 
	nd->arr = d_arr; 
	nd->count = count; 
	
	ndarray *d_nd; cudaMalloc(&d_nd, sizeof(ndarray)); 
	cudaMemcpy(d_nd, nd, sizeof(ndarray), cudaMemcpyHostToDevice); 
	free(nd); 
	
	return d_nd; 
}

ndarray* ndarray_pad(ndarray* base, int* shape_pad) {
	int *new_shape = malloc(sizeof(int) * base->dim); 
	for (int i = 0; i < base->dim; i++) 
		new_shape[i] = base->shape[i] + 2 * shape_pad[i]; 
	ndarray *new_arr = ndarray_create(base->dim, new_shape);
	
	int *pos = malloc(sizeof(int) * base->dim);
	for (int i = 0; i < base->dim; i++) 
		pos[i] = 0;
	int *actual = malloc(sizeof(int) * base->dim); 
	
	do {
		for (int i = 0; i < base->dim; i++) 
			actual[i] = pos[i] + shape_pad[i];
		ndarray_set_val_list(new_arr, actual, ndarray_get_val_list(base, pos));
	}
	while (ndarray_decimal_count(base->dim, pos, base->shape)); 
	
	free(new_shape); 
	free(pos); 
	free(actual); 
	return new_arr; 
}

ndarray* ndarray_pad_gpu(ndarray* base, int* shape_pad) {
	ndarray *padded_nd = ndarray_pad(base, shape_pad); 
	
	int *d_shape; cudaMalloc(&d_shape, sizeof(int) * padded_nd->dim); 
	cudaMemcpy(d_shape, padded_nd->shape, sizeof(int) * padded_nd->dim, cudaMemcpyHostToDevice); 
	
	int *d_cumulative; cudaMalloc(&d_cumulative, sizeof(int) * padded_nd->dim); 
	cudaMemcpy(d_cumulative, padded_nd->cumulative, sizeof(int) * padded_nd->dim, cudaMemcpyHostToDevice); 
	
	ND_TYPE *d_arr; cudaMalloc(&d_arr, sizeof(ND_TYPE) * padded_nd->count); 
	cudaMemcpy(d_arr, padded_nd->arr, sizeof(ND_TYPE) * padded_nd->count, cudaMemcpyHostToDevice);

	ndarray *padded = malloc(sizeof(ndarray)); 
	padded->dim = padded_nd->dim; 
	padded->shape = d_shape; 
	padded->count = padded_nd->count; 
	padded->cumulative = d_cumulative; 
	padded->arr = d_arr; 
	
	ndarray *d_padded; cudaMalloc(&d_padded, sizeof(ndarray)); 
	cudaMemcpy(d_padded, padded, sizeof(ndarray), cudaMemcpyHostToDevice); 
	
	ndarray_free(padded_nd); 
	free(padded); 
	return d_padded; 
}

void ndarray_free(ndarray* nd) {
	free(nd->arr); 
	free(nd->shape); 
	free(nd->cumulative);
	free(nd); 
}

void ndarray_free_gpu(ndarray* nd) {
	cudaFree(nd->arr); 
	cudaFree(nd->shape); 
	cudaFree(nd->cumulative); 
	cudaFree(nd); 
}

ND_TYPE ndarray_get_val_list(ndarray* nd, int* pos) {
	int index = 0;
	for (int i = 0; i < nd->dim; i++) 
		index += nd->cumulative[i] * pos[i]; 
	return nd->arr[index]; 
}

ND_TYPE ndarray_get_val_param(ndarray* nd, ...) {
	va_list valist; 
	va_start(valist, nd); 
	
	int *pos = malloc(sizeof(int) * nd->dim);
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = va_arg(valist, int); 
	
	va_end(valist); 
	ND_TYPE val = ndarray_get_val_list(nd, pos); 
	free(pos); 
	return val;
}

void ndarray_set_val_list(ndarray* nd, int* pos, ND_TYPE value) {
	int index = 0; 
	for (int i = 0; i < nd->dim; i++) 
		index += nd->cumulative[i] * pos[i]; 
	nd->arr[index] = value;
}

void ndarray_set_val_param(ndarray* nd, ND_TYPE value, ...) {
	va_list valist; 
	va_start(valist, value); 
	
	int *pos = malloc(sizeof(int) * nd->dim); 
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = va_arg(valist, int); 
	
	va_end(valist); 
	ndarray_set_val_list(nd, pos, value);
	free(pos); 
}

void ndarray_deep_display(ndarray* nd) {
	int *pos = malloc(sizeof(int) * nd->dim);
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = 0;
	
	int index = 0; 
	do {
		printf("nd"); 
		for (int i = 0; i < nd->dim; i++) { 
			printf("[%d]", pos[i]);
		}			
		printf(" = "); 
		printf(ND_DISPLAY, nd->arr[index++]); 
		printf("\n");
	}
	while (ndarray_decimal_count(nd->dim, pos, nd->shape)); 
	free(pos); 
}

void ndarray_log(ndarray* nd, char* file_name) {
	FILE *fp = fopen(file_name, "w"); 
	if (fp == NULL) {
		fprintf(stderr, "Cannot open file %s\n", file_name); 
		exit(1);
	}		
	
	for (int i = 0; i < nd->dim; i++) 
		fprintf(fp, "%d ", nd->shape[i]);
	fprintf(fp, "\n"); 
	
	for (int i = 0; i < nd->count; i++) 
		fprintf(fp, "%f ", nd->arr[i]); // TODO add ND_DISPLAY 
	
	fclose(fp); 
}

int ndarray_decimal_count(int length, int* counter, int* shape) {
	int i = length-1;
	counter[i]++; 
	while (counter[i] == shape[i]) {
		counter[i] = 0; 
		i--; 
		if (i < 0) return 0; 
		counter[i]++; 
	}
	return 1; 
}

void ndarray_fprint(ndarray* arr, FILE* file) {
	fprintf(file, "[");
	for (int i = 0; i < arr->dim-1; i++) 
		fprintf(file, "%d, ", arr->shape[i]);
	fprintf(file, "%d]", arr->shape[arr->dim-1]); 
}