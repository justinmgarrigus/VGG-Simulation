#include <stdio.h>
#include <string> 
#include "cudaTensorCoreGemm.cuh" 
#include "io.h" 

int main(int argc, char** argv) {
	if (argc != 4) {
		printf("Error: unknown arguments\n"); 
		printf("Usage: ./gemm <inputN_x.bin> <inputN_w.bin> <inputN_b.bin>\n"); 
		exit(1); 
	}
	
	std::string input_file_name  = argv[1]; 
	std::string weight_file_name = argv[2]; 
	std::string bias_file_name   = argv[3]; 
	
	RawDataset<half> inputs(input_file_name); 
	RawDataset<half> weights(weight_file_name); 
	RawDataset<float> biases(bias_file_name); 
	
	int m_global = inputs.rows(),
	    k_global = inputs.cols(), 
	    n_global = weights.cols();

	int m_tiles = m_global / M,
		k_tiles = k_global / K, 
		n_tiles = n_global / N;
	
	half*  A = inputs.data(); 
	half*  B = weights.data(); 
	float* C = biases.data();
	float* D = (float*)malloc(sizeof(float) * m_global * n_global); 
	
	half* d_A; checkCudaErrors(cudaMalloc(&d_A, sizeof(half) * m_global * k_global));
    half* d_B; checkCudaErrors(cudaMalloc(&d_B, sizeof(half) * k_global * n_global));
    float* d_C; checkCudaErrors(cudaMalloc(&d_C, sizeof(float) * m_global * n_global));
    float* d_D; checkCudaErrors(cudaMalloc(&d_D, sizeof(float) * m_global * n_global));

    checkCudaErrors(cudaMemcpy(d_A, A, sizeof(half) * m_global * k_global, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, sizeof(half) * k_global * n_global, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, C, sizeof(float) * m_global * n_global, cudaMemcpyHostToDevice));
	
	enum {
        SHMEM_SZ = MAX(
            sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
            M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	
	printf("compute_gemm(A=%s, B=%s, C=%s)\n", argv[1], argv[2], argv[3]);
	checkKernelErrors((compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(d_A, d_B, d_C, d_D, m_tiles, n_tiles, k_tiles)));
    checkCudaErrors(cudaDeviceSynchronize());
}