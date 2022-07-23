#ifndef CUDA_TENSOR_CORE_GEMM
#define CUDA_TENSOR_CORE_GEMM

#include <cuda_fp16.h> 

extern "C" {
	#include "ndarray.h" 
	
	__host__ void matrix_multiply(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D); 
	int invoke_beginning(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D);
}

//__global__ void compute_gemm(const half* A, const half* B, const float* C, float* D, int m_len, int n_len, int k_len); 

#endif 