#ifndef NDARRAY_H
#define NDARRAY_H 

#define ND_TYPE float
#define ND_DISPLAY "%.1f"

#include "cuda_runtime.h" 

typedef struct ndarray {
	int dim; 
	int *shape; 
	int count; 
	int *cumulative; 
	ND_TYPE *arr; 
} ndarray; 

ndarray* ndarray_create(int count, int* shape);
ndarray* ndarray_create_gpu(int count, int* shape); 
ndarray* ndarray_pad(ndarray* base, int* shape_pad); 
ndarray* ndarray_pad_gpu(ndarray* base, int* shape_pad); 
void ndarray_free(ndarray* nd);
void ndarray_free_gpu(ndarray* nd); 
ndarray* ndarray_copy(ndarray* base, enum cudaMemcpyKind kind); 
ND_TYPE ndarray_get_val_list(ndarray* nd, int* pos); 
ND_TYPE ndarray_get_val_param(ndarray* nd, ...);
void ndarray_set_val_list(ndarray* nd, int* pos, ND_TYPE value);
void ndarray_set_val_param(ndarray* nd, ND_TYPE value, ...);
void ndarray_deep_display(ndarray* nd); 
void ndarray_log(ndarray* nd, char* file_name);
int ndarray_decimal_count(int length, int* counter, int* shape);
void ndarray_fprint(ndarray* arr, FILE* file); 

#endif 