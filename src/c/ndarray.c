#include <stdio.h>
#include <stdlib.h> 
#include <stdarg.h> 
#include <string.h> 
#include "ndarray.h" 

ndarray* ndarray_create(int count, int* shape) {
	int *cumulative = malloc(sizeof(int) * count);
	cumulative[0] = shape[0]; 
	for (int i = 1; i < count; i++) 
		cumulative[i] = cumulative[i-1] * shape[i]; 
	
	ND_TYPE *arr = malloc(sizeof(ND_TYPE) * cumulative[count-1]);
	for (int i = 0; i < cumulative[count-1]; i++) 
		arr[i] = i;
	int size = sizeof(ND_TYPE); 
	
	void *data = arr; 
	void *result = arr; 
	for (int dim = count-1; dim > 0; dim--) {
		result = malloc(sizeof(void*) * cumulative[dim-1]); 
		int index = 0; 
		for (int row = 0; row < cumulative[dim]; row += shape[dim]) {
			((void**)result)[index++] = data + size * row;
		}
		data = result; 
		size = sizeof(void*);
	}
	
	ndarray *nd = malloc(sizeof(ndarray)); 
	nd->dim = count; 
	nd->shape = malloc(sizeof(int) * count); 
	memcpy(nd->shape, shape, sizeof(int) * count); 
	nd->cumulative = cumulative; 
	nd->arr = result; 
	
	return nd; 
}

void ndarray_free(ndarray* nd) {
	void *ptr = nd->arr; 
	for (int i = 0; i < nd->dim; i++) {
		void *next = *((void**)ptr);
		free(ptr); 
		ptr = next; 
	}
	free(nd->shape);
	free(nd->cumulative); 
	free(nd);
}

ND_TYPE ndarray_get_val_list(ndarray* nd, int* pos) {
	void *result = nd->arr; 
	for (int i = 0; i < nd->dim - 1; i++)
		result = *((void**)result + pos[i]);
	return *((ND_TYPE*)result + pos[nd->dim-1]); 
}

ND_TYPE ndarray_get_val_param(ndarray* nd, ...) {
	va_list valist; 
	va_start(valist, nd); 
	
	int pos[nd->dim]; 
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = va_arg(valist, int); 
	
	va_end(valist); 
	return ndarray_get_val_list(nd, pos);
}

void ndarray_set_val_list(ndarray* nd, int* pos, ND_TYPE value) {
	void *result = nd->arr; 
	for (int i = 0; i < nd->dim - 1; i++) 
		result = *((void**)result + pos[i]); 
	*((ND_TYPE*)result + pos[nd->dim-1]) = value; 
}

void ndarray_set_val_param(ndarray* nd, ND_TYPE value, ...) {
	va_list valist; 
	va_start(valist, value); 
	
	int pos[nd->dim]; 
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = va_arg(valist, int); 
	
	va_end(valist); 
	ndarray_set_val_list(nd, pos, value);
}

void ndarray_deep_display(ndarray* nd) {
	int *pos = malloc(sizeof(int) * nd->dim);
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = 0;
	
	do {
		printf("nd"); 
		for (int i = 0; i < nd->dim; i++) { 
			printf("[%d]", pos[i]);
		}			
		printf(" = "); 
		printf(ND_DISPLAY, ndarray_get_val_list(nd, pos)); 
		printf("\n");
	}
	while (ndarray_decimal_count(nd->dim, pos, nd->shape)); 
	free(pos); 
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