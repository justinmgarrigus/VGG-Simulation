#include <stdio.h>
#include <stdlib.h> 
#include <stdarg.h> 
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
	nd->shape = shape; 
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
	free(nd->cumulative); 
	free(nd);
}

ND_TYPE ndarray_val_list(ndarray* nd, int* pos) {
	void *result = nd->arr; 
	for (int i = 0; i < nd->dim - 1; i++)
		result = *((void**)result + pos[i]);
	return *((ND_TYPE*)result + pos[nd->dim-1]); 
}

ND_TYPE ndarray_val_param(ndarray* nd, ...) {
	va_list valist; 
	va_start(valist, nd); 
	
	int pos[nd->dim]; 
	for (int i = 0; i < nd->dim; i++) 
		pos[i] = va_arg(valist, int); 
	
	va_end(valist); 
	return ndarray_val_list(nd, pos);
}

void ndarray_deep_display(ndarray* nd) {
	int *pos = malloc(sizeof(int) * nd->dim); 
	while (pos[0] < nd->shape[0]) {
		printf("nd"); 
		for (int i = 0; i < nd->dim; i++) { 
			printf("[");  
			printf(ND_DISPLAY, pos[i]); 
			printf("]");
		}			
		printf(" = "); 
		printf(ND_DISPLAY, ndarray_val_list(nd, pos)); 
		printf("\n"); 
		
		int index = nd->dim - 1; 
		pos[index]++; 
		while (pos[index] == nd->shape[index] && index > 0) {
			pos[index] = 0; 
			index--; 
			pos[index]++; 
		}
	}
}