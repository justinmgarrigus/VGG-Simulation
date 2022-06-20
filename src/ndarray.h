#define ND_TYPE int 
#define ND_DISPLAY "%d"

#ifndef NDARRAY_H
#define NDARRAY_H 

typedef struct ndarray {
	int dim; 
	int *shape; 
	int *cumulative; 
	void *arr; 
} ndarray; 

ndarray *ndarray_create(int count, int* shape); 
void ndarray_free(ndarray* nd);
ND_TYPE ndarray_val_list(ndarray* nd, int* pos); 
ND_TYPE ndarray_val_param(ndarray* nd, ...); 
void ndarray_deep_display(ndarray* nd); 

#endif 