
/*!
 * Auxillary header file containing CUDA kernel function prototypes and key kernel
 * parameters
 */

#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Number of threads per threadblock to be used for portion of program which
 * sorts exclusively using CUDA/HIP shared memory.
 */
#define THREADS_PER_BLOCK_SHARED_MEM_SORT 256
/*
 * How much shared memory is used for each threadblock for the portion of
 * program which sorts exclusively using shared memory.
 */
#define TILE_SIZE_SHARED_MEM_SORT THREADS_PER_BLOCK_SHARED_MEM_SORT * 8
/*
 * Number of threads per threadblock used for portion of program which sorts
 * using both global and shared memory.
 */
#define THREADS_PER_BLOCK_GLOBAL_MEM_SORT THREADS_PER_BLOCK_SHARED_MEM_SORT / 4
/*
 * Amount of shared memory per threadblock used as linear caching buffer in
 * portion of program which sorts using both global and shared memory
 */
#define TILE_SIZE_GLOBAL_MEM_SORT (THREADS_PER_BLOCK_GLOBAL_MEM_SORT) * 8
/*
 * Any macro definition below this comment will likely never be
 * changed frequently if at all.
 */
/*
 * I genuinely forgot which GitHub repo where I got this following useful macro
 * from but not my original work :/
 */
#define CHECK(call)                                                       \
  {                                                                       \
    cudaError_t err;                                                      \
    if ((err = (call)) != cudaSuccess) {                                  \
      fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                        \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

__global__ void shared_mem_mergesort_subarrays(double* input_array,
                                               double* output_array,
                                               long array_len,
                                               long buff_size,
                                               bool sort_non_descending);

__global__ void global_mem_mergesort_step(double* aux_array,
                                          double* array,
                                          long array_len,
                                          long current_sorted_size,
                                          long max_blocks_per_virtual_grid,
                                          long buff_size,
                                          bool sort_non_descending);

__device__ void global_parallel_merge_using_corank(double* first_input_array,
                                                   double* second_input_array,
                                                   double* dest_array,
                                                   long first_input_array_len,
                                                   long second_input_array_len,
                                                   double* first_array_buff,
                                                   double* second_array_buff,
                                                   long* corank_broadcast_buff,
                                                   long data_buff_len_per_arr,
                                                   long max_num_blocks_per_virtual_grid,
                                                        long num_blocks_this_virtual_grid,
                                                        bool sort_non_descending);

__device__ long determine_corank(long dest_arr_starting_index,
                                                 double *first_input_arr,
                                                 long first_input_arr_len,
                                                 double *second_input_arr,
                                                 long second_input_arr_len,
                                                 bool sort_non_descending);

