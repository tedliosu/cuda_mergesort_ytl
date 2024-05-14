
/*!
 * Unit testing for cuda kernels from main cuda mergesort program
 */

#include <stdlib.h>
#include <math.h>
#include <criterion/parameterized.h>
#include <criterion/criterion.h>
extern "C" {
     #include "main.h"
     #include "splitmix64.h"
     #include "xoshiro256starstar.h"
}
#include "iterative_cuda_mergesort_ytl.h"
#define RAND_NUM_SEED 38972
#define APPEND_NEXT_SMALLEST_FLAG 1
#define NUM_APPEND_OPTS 2
#define SORT_NON_DESC_FLAG 1
#define NO_FAILURE_IDX_FLAG -1
#define MAX_NUM_MERGE_PHASES_PER_TBLOCK_TESTS 32

struct glob_merg_step_kern_param_tupl {

    long sorted_subarr_init_max_size;
    long total_array_size;
    bool sort_non_descending;

};
struct shared_merg_step_kern_param_tupl {

    long arr_len;
    bool sort_non_descending;

};
static double *test_input_array = NULL;
static double *test_output_array = NULL;
static double *test_input_array_gpu = NULL;
static double *test_output_array_gpu = NULL;
static long sorted_subarr_init_sizes[] =
         {1, 2, 3, 37,
        257, 512, 8209, 32771,
                        65536};
static long arr_nums_sorted_subarrays[] =
         {1, 2, 3, 36, 257, 1024};
static bool sort_non_desc_arr[] =
         {true, false};
static long shared_mem_only_sort_total_arr_lens[] =
         {1, 2, 3, 11, 29, 89, 233,
       1031, 2048, 3001, 4099, 6151, 16411, 65537,
         524309, 2097169, 33554467, 134217728};

int compare_longs(const void* first_long, const void* second_long) {
    long diff = *(long *)first_long - *(long *)second_long;
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}

int compare_doubles(const void* first_double, const void* second_double) {
    double diff = *(double *)first_double - *(double *)second_double;
    if (diff < 0.0) {
        return -1;
    } else if (diff > 0.0) {
        return 1;
    } else {
        return 0;
    }
}

int compare_doubles_non_asc(const void* first_double, const void* second_double) {
    double diff = *(double *)second_double - *(double *)first_double;
    if (diff < 0.0) {
        return -1;
    } else if (diff > 0.0) {
        return 1;
    } else {
        return 0;
    }
}

bool is_sorted_properly(double *output_arr, long output_arr_len,
                                long max_sorted_subarr_size, bool sort_non_descending) {

    bool sorted_properly = true;
    long sorted_remainder_subarr_len = output_arr_len % max_sorted_subarr_size;
    long num_sorted_subarrs = output_arr_len / max_sorted_subarr_size;
    long failure_idx_output_arr = NO_FAILURE_IDX_FLAG;
    
    for (long subarr_id = 0; subarr_id < num_sorted_subarrs; ++subarr_id) {
        
        for (long index_in_subarr = 1; index_in_subarr < max_sorted_subarr_size; ++index_in_subarr) {

            long mapped_out_arr_idx = subarr_id * max_sorted_subarr_size + index_in_subarr;

            if ((sort_non_descending && output_arr[mapped_out_arr_idx - 1] > output_arr[mapped_out_arr_idx]) ||
                  (!sort_non_descending && output_arr[mapped_out_arr_idx - 1] < output_arr[mapped_out_arr_idx])) {
                sorted_properly = false;
                failure_idx_output_arr = mapped_out_arr_idx;
                break;
            }

        }

    }

    if (sorted_properly && sorted_remainder_subarr_len > 0) {
        
        for (long mapped_out_arr_idx = output_arr_len - sorted_remainder_subarr_len + 1;
                             mapped_out_arr_idx < output_arr_len; ++mapped_out_arr_idx) {
 
            if ((sort_non_descending && output_arr[mapped_out_arr_idx - 1] > output_arr[mapped_out_arr_idx]) ||
                  (!sort_non_descending && output_arr[mapped_out_arr_idx - 1] < output_arr[mapped_out_arr_idx])) {
                sorted_properly = false;
                failure_idx_output_arr = mapped_out_arr_idx;
                break;
            }

        }

    }

    if (failure_idx_output_arr != NO_FAILURE_IDX_FLAG) {
        printf("Check failed at index %ld in parameter \"output_arr\".", failure_idx_output_arr);
    }

    return sorted_properly;

}

void setup_dynamic_arrays_n_prng(long input_arr_size) {

    set_seed_splitmix64(RAND_NUM_SEED);
    init_xoshiro256starstar();
    test_input_array = (double *) malloc(sizeof(*test_input_array) * input_arr_size);
    test_output_array = (double *) malloc(sizeof(*test_output_array) * input_arr_size);
    CHECK(cudaMalloc(&test_input_array_gpu, sizeof(*test_input_array_gpu) * input_arr_size));
    CHECK(cudaMalloc(&test_output_array_gpu, sizeof(*test_output_array_gpu) * input_arr_size));

}

void fill_array_with_prng(double *input_arr, long input_arr_len) {

    for (long in_arr_idx = 0; in_arr_idx < input_arr_len; ++in_arr_idx) {
        double temp = u64_to_double_conv(xoshiro256starstar_get_next());
        if (isnan(temp)) {
            temp = INFINITY;
        }
        input_arr[in_arr_idx] = temp;
    }

}

void setup_glob_merg_step_kern_tests(void) {
    
    const long arr_of_sortd_subarr_sizes_len = sizeof(sorted_subarr_init_sizes) /
                                                      sizeof(*sorted_subarr_init_sizes);
    const long arr_of_nums_sorted_subarrs_len = sizeof(arr_nums_sorted_subarrays) /
                                                      sizeof(*arr_nums_sorted_subarrays);
    long max_input_array_size = 0;

    /*
     * The following two qsorts may not even be necessary, but unfortunately I don't know
     * when ParameterizedTestParameters executes for a parameterized test relative to
     * this setup. :/
     */
    qsort(sorted_subarr_init_sizes, arr_of_sortd_subarr_sizes_len,
                    sizeof(*sorted_subarr_init_sizes), compare_longs);
    qsort(arr_nums_sorted_subarrays, arr_of_nums_sorted_subarrs_len,
                    sizeof(*arr_nums_sorted_subarrays), compare_longs);

    max_input_array_size = sorted_subarr_init_sizes[arr_of_sortd_subarr_sizes_len - 1] *
                             arr_nums_sorted_subarrays[arr_of_nums_sorted_subarrs_len - 1] +
                             (arr_of_sortd_subarr_sizes_len <= 1 ? 0 : 
                                sorted_subarr_init_sizes[arr_of_sortd_subarr_sizes_len - 2]);

    setup_dynamic_arrays_n_prng(max_input_array_size);

}

void setup_shared_merg_step_kern_tests(void) {

    const long arr_len_shared_mem_only_sort_total_arr_lens = sizeof(shared_mem_only_sort_total_arr_lens) /
                                                                sizeof(*shared_mem_only_sort_total_arr_lens);
    long max_input_array_size = 0;

    qsort(shared_mem_only_sort_total_arr_lens, arr_len_shared_mem_only_sort_total_arr_lens,
                                 sizeof(*shared_mem_only_sort_total_arr_lens), compare_longs);
    max_input_array_size = shared_mem_only_sort_total_arr_lens[arr_len_shared_mem_only_sort_total_arr_lens - 1];

    setup_dynamic_arrays_n_prng(max_input_array_size);

}

void teardown_step_kerns_tests(void) {

    free(test_input_array);
    free(test_output_array);
    CHECK(cudaFree(test_input_array_gpu));
    CHECK(cudaFree(test_output_array_gpu));

}

void free_step_kerns_params(struct criterion_test_params *crp) {

    cr_free(crp->params);

}

ParameterizedTestParameters(params, glob_merg_step_kern_tests) {

    long arr_len_sorted_subarray_sizes = sizeof(sorted_subarr_init_sizes) /
                                           sizeof(*sorted_subarr_init_sizes);
    long arr_len_num_sorted_subarrays = sizeof(arr_nums_sorted_subarrays) /
                                           sizeof(*arr_nums_sorted_subarrays);
    long arr_len_sort_non_descending = sizeof(sort_non_desc_arr) / sizeof (*sort_non_desc_arr);
    const long num_tups = arr_len_sorted_subarray_sizes * arr_len_num_sorted_subarrays *
                                NUM_APPEND_OPTS * arr_len_sort_non_descending;

    struct glob_merg_step_kern_param_tupl *test_params =
        (struct glob_merg_step_kern_param_tupl*) cr_malloc(sizeof(struct glob_merg_step_kern_param_tupl) * num_tups);
    qsort(sorted_subarr_init_sizes, arr_len_sorted_subarray_sizes,
                    sizeof(*sorted_subarr_init_sizes), compare_longs);
    qsort(arr_nums_sorted_subarrays, arr_len_num_sorted_subarrays,
                    sizeof(*arr_nums_sorted_subarrays), compare_longs);

    for (long index_sort_non_desc_arr = 0;
            index_sort_non_desc_arr < arr_len_sort_non_descending; ++index_sort_non_desc_arr) {

        for (long index_arr_appnd_nxt_smallest_subarr = 0;
                index_arr_appnd_nxt_smallest_subarr < NUM_APPEND_OPTS;
                ++index_arr_appnd_nxt_smallest_subarr) {

            for (long index_arr_num_sorted_subarrs = 0;
                    index_arr_num_sorted_subarrs < arr_len_num_sorted_subarrays; ++index_arr_num_sorted_subarrs) {

                for (long index_arr_sorted_subarr_sizes = 0;
                        index_arr_sorted_subarr_sizes < arr_len_sorted_subarray_sizes;
                        ++index_arr_sorted_subarr_sizes) {

                    long param_idx = index_arr_sorted_subarr_sizes +
                                      index_arr_num_sorted_subarrs * arr_len_sorted_subarray_sizes + 
                                      index_arr_appnd_nxt_smallest_subarr * arr_len_num_sorted_subarrays *
                                                                                  arr_len_sorted_subarray_sizes +
                                      index_sort_non_desc_arr * NUM_APPEND_OPTS *
                                                       arr_len_num_sorted_subarrays * arr_len_sorted_subarray_sizes;

                    test_params[param_idx].sorted_subarr_init_max_size = sorted_subarr_init_sizes[index_arr_sorted_subarr_sizes];
                    test_params[param_idx].total_array_size = sorted_subarr_init_sizes[index_arr_sorted_subarr_sizes] *
                                                              arr_nums_sorted_subarrays[index_arr_num_sorted_subarrs];
                    test_params[param_idx].sort_non_descending = sort_non_desc_arr[index_sort_non_desc_arr];

                    if (index_arr_appnd_nxt_smallest_subarr == APPEND_NEXT_SMALLEST_FLAG) {
                        test_params[param_idx].total_array_size += index_arr_sorted_subarr_sizes <= 0 ?
                                                                0 : sorted_subarr_init_sizes[index_arr_sorted_subarr_sizes - 1];
                    }

                }

            }

        }

    }

    return cr_make_param_array(struct glob_merg_step_kern_param_tupl, test_params, num_tups, free_step_kerns_params);

}

ParameterizedTestParameters(params, shared_merg_step_kern_tests) {

    long arr_len_shared_mem_only_sort_total_arr_lens = sizeof(shared_mem_only_sort_total_arr_lens) /
                                                                  sizeof(*shared_mem_only_sort_total_arr_lens);
    long arr_len_sort_non_descending = sizeof(sort_non_desc_arr) / sizeof (*sort_non_desc_arr);

    const long num_tups = arr_len_shared_mem_only_sort_total_arr_lens * arr_len_sort_non_descending;

    struct shared_merg_step_kern_param_tupl *test_params =
        (struct shared_merg_step_kern_param_tupl*) cr_malloc(sizeof(struct shared_merg_step_kern_param_tupl) * num_tups);

    for (long index_sort_non_desc_arr = 0;
              index_sort_non_desc_arr < arr_len_sort_non_descending; ++index_sort_non_desc_arr) {

        for (long index_arr_total_arr_lens = 0;
                index_arr_total_arr_lens < arr_len_shared_mem_only_sort_total_arr_lens; ++index_arr_total_arr_lens) {

            long param_idx = index_arr_total_arr_lens + arr_len_shared_mem_only_sort_total_arr_lens * index_sort_non_desc_arr;

            test_params[param_idx].arr_len = shared_mem_only_sort_total_arr_lens[index_arr_total_arr_lens];
            test_params[param_idx].sort_non_descending = sort_non_desc_arr[index_sort_non_desc_arr];

        }

    }

    return cr_make_param_array(struct shared_merg_step_kern_param_tupl, test_params, num_tups, free_step_kerns_params);


}

ParameterizedTest(struct glob_merg_step_kern_param_tupl *test_tupl, params, glob_merg_step_kern_tests,
                   .init = setup_glob_merg_step_kern_tests, .fini = teardown_step_kerns_tests) {

    printf("Started running sub-test of \"glob_merg_step_kern_tests\" with parameters:\n\t"
            "\"total_array_size\": %ld\n\t"
            "\"sorted_subarr_init_max_size\": %ld\n\t",
            test_tupl->total_array_size, test_tupl->sorted_subarr_init_max_size);
    printf("\"sort_non_descending\": %s\n", test_tupl->sort_non_descending ? "True" : "False");

    long sorted_remainder_subarr_len = test_tupl->total_array_size % test_tupl->sorted_subarr_init_max_size;
    long num_sorted_subarrs = test_tupl->total_array_size / test_tupl->sorted_subarr_init_max_size;

    fill_array_with_prng(test_input_array, test_tupl->total_array_size);

    for (long subarr_id = 0; subarr_id < num_sorted_subarrs; ++subarr_id) {

        long begin_sorted_idx = test_tupl->sorted_subarr_init_max_size * subarr_id;

        if (test_tupl->sort_non_descending) {
            qsort(&test_input_array[begin_sorted_idx], test_tupl->sorted_subarr_init_max_size,
                                                     sizeof(*test_input_array), compare_doubles);
        } else {
            qsort(&test_input_array[begin_sorted_idx], test_tupl->sorted_subarr_init_max_size,
                                                     sizeof(*test_input_array), compare_doubles_non_asc);
        }

    }

    if (sorted_remainder_subarr_len > 0) {

        if(test_tupl->sort_non_descending) {
            qsort(&test_input_array[test_tupl->total_array_size - sorted_remainder_subarr_len],
                        sorted_remainder_subarr_len, sizeof(*test_input_array), compare_doubles);
        } else {
            qsort(&test_input_array[test_tupl->total_array_size - sorted_remainder_subarr_len],
                        sorted_remainder_subarr_len, sizeof(*test_input_array), compare_doubles_non_asc);
        }

    }

    CHECK(cudaMemcpy(test_input_array_gpu, test_input_array,
                        test_tupl->total_array_size * sizeof(*test_input_array), cudaMemcpyHostToDevice));

    long global_mem_sort_buff_size_per_arr = TILE_SIZE_GLOBAL_MEM_SORT;
    long arr_len_per_block = global_mem_sort_buff_size_per_arr * MAX_NUM_MERGE_PHASES_PER_TBLOCK_TESTS;
    if (arr_len_per_block > NUM_ARRAYS_PER_MERGE * test_tupl->sorted_subarr_init_max_size) {
        arr_len_per_block = NUM_ARRAYS_PER_MERGE * test_tupl->sorted_subarr_init_max_size;
    }
    printf("\tComputed \"arr_len_per_block\" for this test is %ld\n", arr_len_per_block);
    long max_blocks_per_virtual_grid = (NUM_ARRAYS_PER_MERGE *
                                             test_tupl->sorted_subarr_init_max_size) / arr_len_per_block;
    dim3 block_size_global_mem_sort(THREADS_PER_BLOCK_GLOBAL_MEM_SORT, 1, 1);
    dim3 grid_size_global_mem_sort(1 + ((test_tupl->total_array_size - 1) / arr_len_per_block), 1, 1);
    void *global_mem_sort_kernel_args[] = {&test_input_array_gpu, &test_output_array_gpu, &(test_tupl->total_array_size),
                                                 &(test_tupl->sorted_subarr_init_max_size), &max_blocks_per_virtual_grid,
                                                      &global_mem_sort_buff_size_per_arr, &(test_tupl->sort_non_descending)};
    
    CHECK(cudaLaunchKernel((void*)global_mem_mergesort_step, grid_size_global_mem_sort,
                                   block_size_global_mem_sort, global_mem_sort_kernel_args,
                                   NUM_TILES_GLOBAL_MEM_SORT * global_mem_sort_buff_size_per_arr *
                                                                             sizeof(*test_output_array_gpu), NULL));

    CHECK(cudaMemcpy(test_output_array, test_output_array_gpu,
                        test_tupl->total_array_size * sizeof(*test_output_array_gpu), cudaMemcpyDeviceToHost));


    cr_assert(is_sorted_properly(test_output_array, test_tupl->total_array_size,
                                                      NUM_ARRAYS_PER_MERGE * test_tupl->sorted_subarr_init_max_size,
                                                              test_tupl->sort_non_descending));


}


ParameterizedTest(struct shared_merg_step_kern_param_tupl *test_tupl, params, shared_merg_step_kern_tests,
                   .init = setup_shared_merg_step_kern_tests, .fini = teardown_step_kerns_tests) {

    printf("Started running sub-test of \"shared_merg_step_kern_tests\" with parameters:\n\t"
            "\"arr_len\": %ld\n\t", test_tupl->arr_len);
    printf("\"sort_non_descending\": %s\n", test_tupl->sort_non_descending ? "True" : "False");

    fill_array_with_prng(test_input_array, test_tupl->arr_len);

    CHECK(cudaMemcpy(test_input_array_gpu, test_input_array,
                        test_tupl->arr_len * sizeof(*test_input_array), cudaMemcpyHostToDevice));
    
    dim3 block_size_shared_mem_sort(THREADS_PER_BLOCK_SHARED_MEM_SORT, 1, 1);
    dim3 grid_size_shared_mem_sort(1 + ((test_tupl->arr_len - 1) / (TILE_SIZE_SHARED_MEM_SORT)), 1, 1);
    long shared_mem_sort_buff_size = TILE_SIZE_SHARED_MEM_SORT;
    void *shared_mem_sort_kernel_args[] = {&test_input_array_gpu, &test_output_array_gpu,
                                             &(test_tupl->arr_len), &shared_mem_sort_buff_size,
                                                                        &(test_tupl->sort_non_descending)};

    CHECK(cudaLaunchKernel((void*)shared_mem_mergesort_subarrays, grid_size_shared_mem_sort,
                              block_size_shared_mem_sort, shared_mem_sort_kernel_args,
                                 NUM_ARRAYS_PER_MERGE * shared_mem_sort_buff_size *
                                                     sizeof(*test_output_array_gpu), NULL));

    CHECK(cudaMemcpy(test_output_array, test_output_array_gpu,
                        test_tupl->arr_len * sizeof(*test_output_array), cudaMemcpyDeviceToHost));

    cr_assert(is_sorted_properly(test_output_array, test_tupl->arr_len,
                                                      shared_mem_sort_buff_size,
                                                          test_tupl->sort_non_descending));

}


