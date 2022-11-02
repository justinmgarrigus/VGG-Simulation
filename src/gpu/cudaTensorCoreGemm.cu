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

__host__ void init_host_matrices(half* a, half* b, float* c) {
//    for (int i = 0; i < M_GLOBAL; i++) {
//        for (int j = 0; j < K_GLOBAL; j++) {
//            a[i * K_GLOBAL + j] = (half)(rand() % 3);
//        }
//    }
//
//    for (int i = 0; i < N_GLOBAL; i++) {
//        for (int j = 0; j < K_GLOBAL; j++) {
//            b[i * K_GLOBAL + j] = (half)(rand() % 3);
//        }
//    }
//
//    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
//        c[t] = static_cast<float>(rand() % 3);
//    }
}

__global__ void compute_gemm(const half* A, const half* B, const float* C, float* D, int M_TILES, int N_TILES, int K_TILES) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    const int M_GLOBAL = M * M_TILES;
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
#pragma unroll
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

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm(half* a, half* b, float* c, float* d, int m_ld,
    int n_ld, int k_ld, float alpha, float beta) {
//    // Leading dimensions. Packed with no transpositions.
//    int lda = k_ld;
//    int ldb = k_ld;
//    int ldc = n_ld;
//
//    // Tile using a 2D grid
//    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
//
//    // Declare the fragments
//    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
//        a_frag;
//    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
//        b_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
//    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
//
//    wmma::fill_fragment(acc_frag, 0.0f);
//
//    // Loop over k
//    for (int i = 0; i < k_ld; i += WMMA_K) {
//        int aCol = i;
//        int aRow = warpM * WMMA_M;
//        int bCol = warpN * N;
//        int bRow = i;
//
//        // Bounds checking
//        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
//            // Load the inputs
//            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
//            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
//
//            // Perform the matrix multiplication
//            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//        }
//    }
//
//    // Load in the current value of c, scale it by beta, and add this our result
//    // scaled by alpha
//    int cCol = warpN * WMMA_N;
//    int cRow = warpM * WMMA_M;
//
//    if (cRow < m_ld && cCol < n_ld) {
//        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
//            wmma::mem_row_major);
//
//        for (int i = 0; i < c_frag.num_elements; i++) {
//            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
//        }
//
//        // Store the output
//        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
//            wmma::mem_row_major);
//    }
}

__host__ void matMultiplyOnHost(half* A, half* B, float* C, float alpha,
    float beta, int numARows, int numAColumns,
    int numBRows, int numBColumns, int numCRows,
    int numCColumns) {
//    for (int i = 0; i < numCRows; i++) {
//        for (int j = 0; j < numCColumns; j++) {
//            float temp = 0.0;
//
//            for (int k = 0; k < numAColumns; k++) {
//                temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
//            }
//
//            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
//        }
//    }
}

// multiple * n < value <= multiple * (n + 1), for some n. Returns multiple * (n + 1) 
int pad_multiple(int value, int multiple) {
    return (value / multiple + 1) * multiple;
}

__host__ void matrix_multiply(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D) {
	printf("Matrix multiply\n"); 
	
    // Assumption: since compute_gemm considers B to be col-wise (which doesn't work 
    // for our purposes), we need to transpose it. Thus, B must be square!
	// If these are reshaped, then a_reformatted and b_reformatted will not be null.  
	ndarray *b_reformatted = nullptr;
	ndarray *a_reformatted = nullptr; 
    if (h_B->shape[0] != h_B->shape[1]) {
		if (h_B->shape[0] > h_B->shape[1]) {
			// Resize B to be square based on the size of h_B->shape[0]
			const int input_size = h_A->shape[1];
			const int output_size = h_B->shape[1];
	
			int reformat_shape[2] = { input_size, input_size };
			b_reformatted = ndarray_create(2, reformat_shape);
	
			for (int r = 0; r < input_size; r++) {
				for (int c = 0; c < output_size; c++)
					b_reformatted->arr[r * input_size + c] = h_B->arr[r * output_size + c];
				for (int c = output_size; c < input_size; c++)
					b_reformatted->arr[r * input_size + c] = 0;
			}
			
			h_B = b_reformatted;
		}
		else {
			// Resize B to be square based on the size of h_B->shape[1]
			int b_reformat_shape[2] = { h_B->shape[1], h_B->shape[1] }; 
			b_reformatted = ndarray_create(2, b_reformat_shape);
			
			for (int i = 0; i < h_B->count; i++) 
				b_reformatted->arr[i] = h_B->arr[i]; 
			for (int i = h_B->count; i < b_reformatted->count; i++) 
				b_reformatted->arr[i] = 0;
			
			// Resize A to align with B's new size
			int a_reformat_shape[2] = { h_A->shape[0], h_B->shape[1] };
			a_reformatted = ndarray_create(2, a_reformat_shape);
	
			for (int r = 0; r < h_A->shape[0]; r++) {
				for (int c = 0; c < h_A->shape[1]; c++)
					a_reformatted->arr[r * a_reformatted->shape[1] + c] = h_A->arr[r * h_A->shape[1] + c];
				for (int c = h_A->shape[1]; c < a_reformatted->shape[1]; c++)
					a_reformatted->arr[r * a_reformatted->shape[1] + c] = 0;
			}
			
			h_B = b_reformatted;
			h_A = a_reformatted;		
		}
    }

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
    int m_padding = m_tiles - m_shape,
        k_padding = k_tiles - k_shape,
        n_padding = n_tiles - n_shape;
		
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
			B[i * k_global + j] = (half)h_B->arr[j * h_B->shape[1] + i];
		for (int j = h_B->shape[1]; j < k_global; j++) 
			B[i * k_global + j] = (half)0.0f; 
	}
	for (int i = h_B->shape[0]; i < n_global; i++) 
		for (int j = 0; j < k_global; j++) 
			B[i * k_global + j] = (half)0.0f;
	
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

    if (b_reformatted != nullptr) ndarray_free(b_reformatted);
	if (a_reformatted != nullptr) ndarray_free(a_reformatted); 
}

int invoke_beginning(ndarray* h_A, ndarray* h_B, ndarray* h_C, ndarray* h_D) {
	int m_shape = h_A->shape[0] / M,
        k_shape = h_A->shape[1] / K,
        n_shape = h_B->shape[1] / N;
    
	const int 
		M_TILES = pad_multiple(m_shape, 8),
        K_TILES = pad_multiple(k_shape, 8),
        N_TILES = pad_multiple(n_shape, 8);
	
    const int 
        M_GLOBAL = M_TILES * M,
        N_GLOBAL = N_TILES * N,
        K_GLOBAL = K_TILES * K;

    printf("Initializing...\n");

    int dev = 0; // findCudaDevice(argc, (const char**)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

//    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
//    if (deviceProp.major < 7) {
//        printf(
//            "cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
//            "Cores.  Exiting...\n");
//        exit(EXIT_WAIVED);
//    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    half* A_h = NULL;
    half* B_h = NULL;
    float* C_h = NULL;
#if CPU_DEBUG
    float* result_hD = NULL;
    float* result_host = NULL;
#endif

    A_h = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
    result_hD = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    result_host = (float*)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

    half* A = NULL;
    half* B = NULL;
    float* C = NULL;
    float* D = NULL;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&A),
        sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&B),
        sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&C),
        sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&D),
        sizeof(float) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    printf("Preparing data for GPU...\n");

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

//	half *A; cudaMalloc(&A, sizeof(half) * M_GLOBAL * K_GLOBAL); 
	
    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
        // per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the
        // maximum of those
        // two numbers.
        SHMEM_SZ = MAX(
            sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
            M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
            (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // If enough shared memory available on the GPU use high performant kernel
    if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
        printf("Computing... using high performance kernel compute_gemm \n");

        printf("<<<%d, %d, %d>>>\n", deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ);
        checkCudaErrors(cudaFuncSetAttribute(
            compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
        checkKernelErrors(
            (compute_gemm << <deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                SHMEM_SZ >> > (A, B, C, D, M_TILES, N_TILES, K_TILES)));
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D,
            sizeof(float) * M_GLOBAL * N_GLOBAL,
            cudaMemcpyDeviceToHost));
#endif
    }
    else {
//        dim3 gridDim;
//        dim3 blockDim;
//
//        // blockDim.x must be a multple of warpSize
//        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
//        blockDim.x = 128;
//        blockDim.y = 4;
//
//        gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
//            (WMMA_M * blockDim.x / 32);
//        gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
//
//        printf("Computing... using simple_wmma_gemm kernel\n");
//        simple_wmma_gemm << <gridDim, blockDim >> > (A, B, C, D, M_GLOBAL, N_GLOBAL,
//           K_GLOBAL, alpha, beta);
//#if CPU_DEBUG
//        checkCudaErrors(cudaMemcpy(result_hD, D,
//            sizeof(float) * M_GLOBAL * N_GLOBAL,
//            cudaMemcpyDeviceToHost));
//#endif
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

    float temp = 0;
    for (int k = 0; k < N_GLOBAL; k++) {
        // temp += (float)A_h[k] * (float)B_h[k]; // This causes an error on Cuda 9.2 
    }
    float result = temp * C_h[0];
    printf("Actual: %f, ours: %f\n", result_hD[0], result);

    //  matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
    //                    K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

    //  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    //    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
    //      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
    //             result_host[i]);
    //  }
    free(result_hD);
    free(result_host);
#endif

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
        N_GLOBAL * K_GLOBAL * 2) /
        (milliseconds / 1000.)) /
        1e12);

    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void*>(D)));

	printf("Returning from invoke_beginning\n"); 
    return 0;
}
