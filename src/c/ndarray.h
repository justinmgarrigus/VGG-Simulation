#define ND_TYPE double
#define ND_DISPLAY "%.1f"

#ifndef NDARRAY_H
#define NDARRAY_H 

typedef struct ndarray {
	int dim; 
	int *shape; 
	int count; 
	int *cumulative; 
	ND_TYPE *arr; 
} ndarray; 

ndarray* ndarray_create(int count, int* shape);
ndarray* ndarray_pad(ndarray* base, int* shape_pad); 
void ndarray_free(ndarray* nd);
ND_TYPE ndarray_get_val_list(ndarray* nd, int* pos); 
ND_TYPE ndarray_get_val_param(ndarray* nd, ...); 
void ndarray_set_val_list(ndarray* nd, int* pos, ND_TYPE value); 
void ndarray_set_val_param(ndarray* nd, ND_TYPE value, ...); 
void ndarray_deep_display(ndarray* nd); 
void ndarray_log(ndarray* nd, char* file_name);
int ndarray_decimal_count(int length, int* counter, int* shape);
void ndarray_fprint(ndarray* arr, FILE* file); 

#endif 