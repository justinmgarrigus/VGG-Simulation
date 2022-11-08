/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 // CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
 // and Accumulate API introduced in CUDA 9.

 // In this program, the compute_gemm kernel computes the result of a matrix
 // multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
 // both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
 // K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
 // that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
 // iteration. When the tile is computed, the CTA stores it to the global memory
 // and begins a new iteration, selecting a new 128 x 128 tile to compute.
 // Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
 // eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
 // compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
 // moving through the K_GLOBAL dimension of the A and B matrices and
 // accumulating the intermediate result in the local thread state.

 // There are a number of simple optimizations used in the algorithm:
 // - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
 //   shared memory. After that is done, each warp loads the C matrix fragments
 //   from shared memory, thus avoiding a random global memory access.
 // - On each internal iteration, the CTA copies a portion of the A and B
 //   matrices from global memory to shared memory. After that, all warps in the
 //   CTA reuse the A and B data from shared memory, thus reducing the number of
 //   data copies from global memory.
 // - The portions of the A and B matrices are stored in shared memory with an
 //   additional padding (skew) to reduce the number of shared memory access bank
 //   conflicts.
 //   (See a detailed explanation near the SKEW_HALF macro definition.)
 // - When the CTA finishes computing the tiles of the resulting matrix, each
 //   warp stores its subtiles to shared memory. The CTA then copies the shared
 //   memory contents to global memory, again avoiding redundant random global
 //   memory  accesses.
 // - Note that the CTA tile size is chosen to maximize the GPU register
 //   utilization, but carefully enough to avoid local memory use.

// TODO: includes were here, but were moved to cudaTensorCoreGemm.cuh to make 
// porting a bit easier. The includes should probably stay in here in the future.
#include "cudaTensorCoreGemm.cuh"

using namespace nvcuda;

#if defined(SAVE_INTERMEDIATE) 
	int conv_layer_counter = 1; 
#endif 

__global__ void compute_gemm(const half* A, const half* B, const float* C, float* D, int M_TILES, int N_TILES, int K_TILES) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    const int N_GLOBAL = N * N_TILES;
    const int K_GLOBAL = K * K_TILES;

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    float* shmem_warp_tile_ptr = (float*)&shmem[0][0] +
        (warpId / 2) * SHMEM_STRIDE * K * 2 +
        (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    float* shmem_warp_stream_ptr =
        (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i =
            ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx =
            (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float* src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;

            *((copy_t*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t*)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                    laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const float* tile_ptr =
                    shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half* warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
            M * K_GLOBAL * (warpId % 4) * 2)
            : (&B[block_tile_j * N * K_GLOBAL] +
                N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx =
                warpId < (WARPS_PER_BLOCK / 2)
                ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4* lane_ptr = (int4*)(warp_ptr + tile_k * K +
                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                i++) {
                // Copy 16 bytes at once in each lane.
                *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                    *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr =
                    (int4*)((half*)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                    a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                    b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half* tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off +
                                (WARP_ROW_TILES * N) * (warpId % 2) +
                                (j * N);
                            const half* tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }
        
            // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                float* tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        float* dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}

#if (0 > 1) // (CUDART_VERSION > 9010) TODO: make this more efficient on newer architectures. 

__host__ void ndarray_to_half_arr(half* A, half* B, ndarray* h_A, ndarray* h_B, 
                                  int m_global, int k_global, int n_global) 
{
	// Copy h_A to A (assuming that it's a multiple of 8)
	for (int i = 0; i < h_A->shape[0]; i++) {
		for (int j = 0; j < h_A->shape[1]; j++)
			A[i * k_global + j] = (half)h_A->arr[i * h_A->shape[1] + j];
		for (int j = h_A->shape[1]; j < k_global; j++) 
			A[i * k_global + j] = (half)0.0f; 
	}
	for (int i = h_A->shape[0]; i < m_global; i++) 
		for (int j = 0; j < k_global; j++) 
			A[i * k_global + j] = (half)0.0f;
    
    // TODO: Since the compute_gemm kernel considers B to be in col-major order, 
    // we have to transpose the matrix when loading it in. This means we have to
    // assume B is square, so enough space is allowed for a transpose. This 
    // should be fixed in the future to conserve space! 
	for (int i = 0; i < h_B->shape[0]; i++) {
		for (int j = 0; j < h_B->shape[1]; j++)
			B[i * k_global + j] = h_B->arr[j * h_B->shape[1] + i];
		for (int j = h_B->shape[1]; j < k_global; j++) 
			B[i * k_global + j] = (half)0.0f; 
	}
	for (int i = h_B->shape[0]; i < n_global; i++) 
		for (int j = 0; j < k_global; j++) 
			B[i * k_global + j] = (half)0.0f;
}

#else 

__global__ void arr_float2half(half* dest, float* src, int count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	if (index > count) return; 
	dest[index] = __float2half(src[index]); 
}

__global__ void transfer_arr(half* dest, float* src, 
							 int dest_rows, int dest_cols, 
							 int src_rows, int src_cols) 
{
	for (int i = 0; i < src_rows; i++) {
		for (int j = 0; j < src_cols; j++)
			dest[i * dest_cols + j] = __float2half(src[i * src_cols + j]); 
		for (int j = src_cols; j < dest_cols; j++) 
			dest[i * dest_cols + j] = 0; 
	}
	for (int i = src_rows; i < dest_rows; i++) 
		for (int j = 0; j < dest_cols; j++) 
			dest[i * dest_cols + j] = 0;
}

// Cuda 9.1 does not support half values at all on the host, so we need to do 
// all conversions on the device. 
__host__ void ndarray_to_half_arr(half* A, half* B, ndarray* h_A, ndarray* h_B, 
                                  int m_global, int k_global, int n_global) 
{
	const int thread_count = 1024; // TODO: this should be variable. 
	
	// Copy h_A and h_B to device
	float *A_arr = nullptr; cudaMalloc(&A_arr, sizeof(float) * h_A->count); 
	float *B_arr = nullptr; cudaMalloc(&B_arr, sizeof(float) * h_B->count); 
	cudaMemcpy(A_arr, h_A->arr, sizeof(float) * h_A->count, cudaMemcpyHostToDevice); 
	cudaMemcpy(B_arr, h_B->arr, sizeof(float) * h_B->count, cudaMemcpyHostToDevice);
	
	// Reformat data to properly align with the gemm function 
	half *A_re = nullptr; cudaMalloc(&A_re, sizeof(half) * m_global * k_global); 
	half *B_re = nullptr; cudaMalloc(&B_re, sizeof(half) * k_global * n_global); 
	transfer_arr<<<1,1>>>(A_re, A_arr, m_global, k_global, h_A->shape[0], h_A->shape[1]);
	transfer_arr<<<1,1>>>(B_re, B_arr, k_global, n_global, h_B->shape[0], h_B->shape[1]);	
						  
	// Copy reformatted data back to goal output 
    cudaMemcpy(A, A_re, sizeof(half) * m_global * k_global, cudaMemcpyDeviceToHost); 
    cudaMemcpy(B, B_re, sizeof(half) * k_global * n_global, cudaMemcpyDeviceToHost); 
    
    // Free all allocated data 
    cudaFree(A_arr); 
    cudaFree(B_arr);
    cudaFree(A_re); 
    cudaFree(B_re); 
}

#endif 

// multiple * n < value <= multiple * (n + 1), for some n. Returns multiple * (n + 1) 
int pad_multiple(int value, int multiple) {
    return (value / multiple + 1) * multiple;
}

__host__ void matrix_multiply(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D) {
	printf("Matrix multiply\n"); 
	
	// Convert h_B from row-wise to col-wise
	int col_wise_shape[2] = { h_B->shape[1], h_B->shape[0] }; 
	ndarray *h_B_col = ndarray_create(2, col_wise_shape); 
	
	for (int i = 0; i < h_B->shape[0]; i++)
		for (int j = 0; j < h_B->shape[1]; j++)
			h_B_col->arr[j * h_B->shape[0] + i] = h_B->arr[i * h_B->shape[1] + j];
			
	h_B = h_B_col;

	enum {
        SHMEM_SZ = MAX(
            sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
            M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    // Round up the dimensions to the nearest multiple of 8.
    int m_shape = h_A->shape[0] / M,
        k_shape = h_A->shape[1] / K,
        n_shape = h_B->shape[1] / N;
    int m_tiles = pad_multiple(m_shape, 8),
        k_tiles = pad_multiple(k_shape, 8),
        n_tiles = pad_multiple(n_shape, 8);
		
    int m_global = M * m_tiles,
        k_global = K * k_tiles,
        n_global = N * n_tiles;
		
	printf("Dimensions: (%d, %d) ; (%d, %d) ; (%d, %d)\n", h_A->shape[0], h_A->shape[1], h_B->shape[0], h_B->shape[1], h_C->shape[0], h_C->shape[1]);
	printf("%d, %d, %d\n", m_global, k_global, n_global); 

	// Convert into equivalent arrays. 
    half* A = (half*)malloc(sizeof(half) * m_global * k_global);
    half* B = (half*)malloc(sizeof(half) * k_global * n_global);
	float* C = (float*)malloc(sizeof(float) * m_global * n_global);
	float* D = (float*)malloc(sizeof(float) * m_global * n_global);

	// Dependent on Cuda version, populate A and B
	ndarray_to_half_arr(A, B, h_A, h_B, m_global, k_global, n_global); 
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error: ndarray_to_half_arr (%s)\n", cudaGetErrorName(cudaGetLastError())); 
		exit(1); 
	}

	// Copy h_C to C (assuming that it's a multiple of 8)
	for (int i = 0; i < h_C->shape[0]; i++) {
		for (int j = 0; j < h_C->shape[1]; j++)
			C[i * n_global + j] = h_C->arr[i * h_C->shape[1] + j];
		for (int j = h_C->shape[1]; j < n_global; j++) 
			C[i * n_global + j] = 0.0f; 
	}
	for (int i = h_C->shape[0]; i < m_global; i++) 
		for (int j = 0; j < n_global; j++) 
			C[i * n_global + j] = 0.0f;
			
	// Save intermediate inputs to a file. 
#if defined(SAVE_INTERMEDIATE)
	std::string file_name = "conv_data/conv" + std::to_string(conv_layer_counter++);
	
	RawDataset<half> input(A, m_global, k_global);
	input.save(file_name + "_x.bin"); 
	
	RawDataset<half> weights(B, k_global, n_global); 
	weights.save(file_name + "_w.bin");
	
	RawDataset<float> bias(C, m_global, n_global); 
	bias.save(file_name + "_b.bin");
#endif 
	
    half* d_A; checkCudaErrors(cudaMalloc(&d_A, sizeof(half) * m_global * k_global));
    half* d_B; checkCudaErrors(cudaMalloc(&d_B, sizeof(half) * k_global * n_global));
    float* d_C; checkCudaErrors(cudaMalloc(&d_C, sizeof(float) * m_global * n_global));
    float* d_D; checkCudaErrors(cudaMalloc(&d_D, sizeof(float) * m_global * n_global));

    checkCudaErrors(cudaMemcpy(d_A, A, sizeof(half) * m_global * k_global, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, sizeof(half) * k_global * n_global, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, C, sizeof(float) * m_global * n_global, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_D, D, sizeof(float) * m_global * n_global, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    
	checkKernelErrors((compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(d_A, d_B, d_C, d_D, m_tiles, n_tiles, k_tiles)));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(D, d_D, sizeof(float) * m_global* n_global, cudaMemcpyDeviceToHost)); 
	for (int i = 0; i < h_D->shape[0]; i++)
		for (int j = 0; j < h_D->shape[1]; j++)
			h_D->arr[i * h_D->shape[1] + j] = D[i * n_global + j]; 
	
    free(A);
    free(B);
    free(C);
    free(D);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_D));
	free(h_B_col); 
}