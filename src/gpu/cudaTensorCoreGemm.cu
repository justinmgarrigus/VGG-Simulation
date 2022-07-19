#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

extern "C" {
	#include "cudaTensorCoreGemm.cuh"
}

#define SHARED_MEMORY_LIMIT_64K 1

// GPU configuration.

#define WARP_SIZE 32

#define CPU_DEBUG 1

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

const int M_TILES = 8; // Must be a multiple of 8  
const int N_TILES = 8; 
const int K_TILES = 8; 

const int M_GLOBAL = (M * M_TILES); 
const int N_GLOBAL = (N * N_TILES); 
const int K_GLOBAL = (K * K_TILES); 

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define CHUNK_K 4

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

#define SKEW_HALF 16

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;

__host__ void init_host_matrices(half *a, half *b, float *c) {
	double counter = 0; 
	for (int i = 0; i < M_GLOBAL; i++) {
		for (int j = 0; j < K_GLOBAL; j++) {
			a[i * K_GLOBAL + j] = (half)(counter / 128.0);
			counter++; 
		}
	}

	counter = 0; 
	for (int i = 0; i < N_GLOBAL; i++) {
		for (int j = 0; j < K_GLOBAL; j++) {
			b[i * K_GLOBAL + j] = (half)(counter / 128.0);
			counter++; 
		}
	}

	for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
		c[t] = 0.0;
	}

	printf("%f, %f\n", (float)a[1], (float)b[1]);
}

__host__ void matrix_multiply(ndarray* d_A, ndarray* d_B, ndarray* d_C, ndarray* d_D) {
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
	
	// Round up the dimensions to the nearest multiple of 8. 
	int k_tiles = (d_A->shape[1] / 8 + 1) * 8; 
	int m_tiles = 8; 
	int n_tiles = (d_B->shape[1] / 8 + 1) * 8;
	
	half *a = NULL; 
	half *b = NULL; 
	float *c = NULL; 
	float *d = NULL; 
	
	printf("We're hre in matrix multiply!\n"); 
	//compute_gemm<<<16, THREADS_PER_BLOCK, SHMEM_SZ>>>(a, b, c, d, m_tiles, n_tiles, k_tiles); 
}

__global__ void compute_gemm(const half *A, const half *B, const float *C, float *D, int m_len, int n_len, int k_len) {
	extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];
	
	const int M_TILES = m_len / 8 + 1; // TODO fix for mod-8 vals 
	const int N_TILES = n_len / 8 + 1; 
	const int K_TILES = k_len / 8 + 1; 
	const int M_GLOBAL = M * M_TILES; 
	const int N_GLOBAL = N * N_TILES; 
	const int K_GLOBAL = K * K_TILES; 

	// Warp and lane identification.
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
															 (warpId / 2) * SHMEM_STRIDE * K * 2 +
															 (warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr =
			(float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

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
		const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

		// Stream multiple C tiles to shared memory.
#pragma unroll
		for (int i = 0; i < K; i++) {
			typedef int4 copy_t;

			*((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
					*((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
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
				const float *tile_ptr =
						shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Select what warp copies what matrix to shared memory.
		// Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
		const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
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
			int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
																(laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
											 (laneId % CHUNK_COPY_LINE_LANES);

			// Shift the second half of the warp to the next row / column in the
			// shared memory.
			shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
			for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
					 i++) {
				// Copy 16 bytes at once in each lane.
				*((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
						*lane_ptr;

				// Advance the global memory pointer and the shared memory index.
				lane_ptr =
						(int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
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
					const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

					wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
					for (int j = 0; j < WARP_ROW_TILES; j++) {
						if (i == 0) {
							// Load the B matrix fragment once, because it is going to be
							// reused against the other A matrix fragments.
							size_t shmem_idx_b = shmem_idx_b_off +
																	 (WARP_ROW_TILES * N) * (warpId % 2) +
																	 (j * N);
							const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

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
				float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

				wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
			}
		}

		__syncthreads();

		// Now that shared memory contains all the D tiles, stream them to global
		// memory.
		float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
		for (int i = 0; i < K; i++) {
			*((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
					*((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
		}

		__syncthreads();
	}
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
	for (int i = 0; i < numCRows; i++) {
		for (int j = 0; j < numCColumns; j++) {
			float temp = 0.0;

			for (int k = 0; k < numAColumns; k++) {
				temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
			}

			C[i * numCColumns + j] = temp + C[i * numCColumns + j];
		}
	}
}

int main_unused(int argc, char **argv) {
	printf("Initializing...\n");

	int dev = findCudaDevice(argc, (const char **)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	// Tensor cores require a GPU of Volta (SM7X) architecture or higher.
	if (deviceProp.major < 7) {
		printf(
				"cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
				"Cores.  Exiting...\n");
		exit(EXIT_WAIVED);
	}

	printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
	printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
	printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

	half *A_h = NULL;
	half *B_h = NULL;
	float *C_h = NULL;
#if CPU_DEBUG
	float *result_hD = NULL;
	float *result_host = NULL;
#endif

	A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
	B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
	C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
	result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
	result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

	half *A = NULL;
	half *B = NULL;
	float *C = NULL;
	float *D = NULL;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A),
														 sizeof(half) * M_GLOBAL * K_GLOBAL));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B),
														 sizeof(half) * N_GLOBAL * K_GLOBAL));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C),
														 sizeof(float) * M_GLOBAL * N_GLOBAL));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D),
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

		checkCudaErrors(cudaFuncSetAttribute(
				compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
		checkKernelErrors(
				(compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
												SHMEM_SZ>>>(A, B, C, D, M_TILES, N_TILES, K_TILES)));
#if CPU_DEBUG
		checkCudaErrors(cudaMemcpy(result_hD, D,
															 sizeof(float) * M_GLOBAL * N_GLOBAL,
															 cudaMemcpyDeviceToHost));
#endif
	} 

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
	printf("Verifying correctness of the computations...\n");

	memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

		float temp = 0; 
		for (int k = 0; k < N_GLOBAL; k++) {
				temp += (float)A_h[k] * (float)B_h[k]; 
		}
		float result = temp + C_h[0]; 
		printf("Actual: %f, ours: %f\n", result_hD[0], result);

	matMultiplyOnHost(A_h, B_h, result_host, M_GLOBAL, K_GLOBAL,
						K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
	
	for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
		if (fabs(result_hD[i] - result_host[i]) > 0.1f)
		printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
				result_host[i]);
	}
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
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

	return 0;
}
