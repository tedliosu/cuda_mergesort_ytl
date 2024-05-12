
/*!
 * Unit testing for cuda kernels from main cuda mergesort program
 */

#include <stdlib.h>
#include <criterion/parameterized.h>
#include <criterion/criterion.h>
extern "C" {
     #include "splitmix64.h"
     #include "xoshiro256starstar.h"
}
#include "iterative_cuda_mergesort_ytl.h"
#define RAND_NUM_SEED 38972
#define APPEND_NEXT_SMALLEST_FLAG 1
#define SORT_NON_DESC_FLAG 1


struct glob_merg_step_kern_param_tupl {

    long sorted_subarray_max_size;
    long total_array_size;
    bool sort_non_descending;

};
typedef struct max_array_n_index_tupl {

    long max_val;
    long index_of_max_val;

} array_max_n_corresp_index_type;
static double *test_input_array = NULL;
static double *test_output_array = NULL;
static double *test_input_array_gpu = NULL;
static double *test_output_array_gpu = NULL;
static long sorted_subarray_sizes[] =
         {1, 2, 3, 37,
        257, 512, 8209, 32771,
                        65536};
static long arr_nums_sorted_subarrays[] =
         {1, 2, 3, 36, 257, 1024};
static bool arr_appnd_nxt_smallest_subarr[] = 
         {true, false};
static bool sort_non_desc_arr[] =
         {true, false};

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

    return sorted_properly;

}

void setup_glob_merg_step_kern_tests(void) {
    
    const long arr_of_sortd_subarr_sizes_len = sizeof(sorted_subarray_sizes) /
                                                      sizeof(*sorted_subarray_sizes);
    const long arr_of_nums_sorted_subarrs_len = sizeof(arr_nums_sorted_subarrays) /
                                                      sizeof(*arr_nums_sorted_subarrays);
    long max_input_array_size = 0;

    set_seed_splitmix64(RAND_NUM_SEED);
    init_xoshiro256starstar();
    qsort(sorted_subarray_sizes, arr_of_sortd_subarr_sizes_len,
                    sizeof(*sorted_subarray_sizes), compare_longs);
    qsort(arr_nums_sorted_subarrays, arr_of_nums_sorted_subarrs_len,
                    sizeof(*arr_nums_sorted_subarrays), compare_longs);

    max_input_array_size = sorted_subarray_sizes[arr_of_sortd_subarr_sizes_len - 1] *
                             arr_nums_sorted_subarrays[arr_of_nums_sorted_subarrs_len - 1] +
                             (arr_of_sortd_subarr_sizes_len <= 1 ? 0 : 
                                sorted_subarray_sizes[arr_of_sortd_subarr_sizes_len - 2]);

    test_input_array = (double *) malloc(sizeof(*test_input_array) * max_input_array_size);
    test_output_array = (double *) malloc(sizeof(*test_output_array) * max_input_array_size);
    CHECK(cudaMalloc(&test_input_array_gpu, sizeof(*test_input_array_gpu) * max_input_array_size));
    CHECK(cudaMalloc(&test_output_array_gpu, sizeof(*test_output_array_gpu) * max_input_array_size));

}

void teardown_glob_merg_step_kern_tests(void) {

    free(test_input_array);
    free(test_output_array);
    CHECK(cudaFree(test_input_array_gpu));
    CHECK(cudaFree(test_output_array_gpu));

}

void free_glob_merg_step_kern_params(struct criterion_test_params *crp) {

    cr_free(crp->params);

}

ParameterizedTestParameters(params, glob_merg_step_kern_tests) {

    long arr_len_sorted_subarray_sizes = sizeof(sorted_subarray_sizes) /
                                           sizeof(*sorted_subarray_sizes);
    long arr_len_num_sorted_subarrays = sizeof(arr_nums_sorted_subarrays) /
                                           sizeof(*arr_nums_sorted_subarrays);
    long arr_len_append_next_smallest_subarray = sizeof(arr_appnd_nxt_smallest_subarr) /
                                                    sizeof (*arr_appnd_nxt_smallest_subarr);
    long arr_len_sort_non_descending = sizeof(sort_non_desc_arr) / sizeof (*sort_non_desc_arr);
    const long num_tups = arr_len_sorted_subarray_sizes * arr_len_num_sorted_subarrays *
                                arr_len_append_next_smallest_subarray * arr_len_sort_non_descending;

    struct glob_merg_step_kern_param_tupl *test_params =
        (struct glob_merg_step_kern_param_tupl*) cr_malloc(sizeof(struct glob_merg_step_kern_param_tupl) * num_tups);
    qsort(sorted_subarray_sizes, arr_len_sorted_subarray_sizes,
                    sizeof(*sorted_subarray_sizes), compare_longs);
    qsort(arr_nums_sorted_subarrays, arr_len_num_sorted_subarrays,
                    sizeof(*arr_nums_sorted_subarrays), compare_longs);

    for (long index_sort_non_desc_arr = 0;
            index_sort_non_desc_arr < arr_len_sort_non_descending; ++index_sort_non_desc_arr) {

        for (long index_arr_appnd_nxt_smallest_subarr = 0;
                index_arr_appnd_nxt_smallest_subarr < arr_len_append_next_smallest_subarray;
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
                                      index_sort_non_desc_arr * arr_len_append_next_smallest_subarray *
                                                       arr_len_num_sorted_subarrays * arr_len_sorted_subarray_sizes;

                    test_params[param_idx].sorted_subarray_max_size = sorted_subarray_sizes[index_arr_sorted_subarr_sizes];
                    test_params[param_idx].total_array_size = sorted_subarray_sizes[index_arr_sorted_subarr_sizes] *
                                                              arr_nums_sorted_subarrays[index_arr_num_sorted_subarrs];
                    test_params[param_idx].sort_non_descending = sort_non_desc_arr[index_sort_non_desc_arr];

                    if (index_arr_appnd_nxt_smallest_subarr == APPEND_NEXT_SMALLEST_FLAG) {
                        test_params[param_idx].total_array_size += index_arr_sorted_subarr_sizes <= 0 ?
                                                                0 : sorted_subarray_sizes[index_arr_sorted_subarr_sizes - 1];
                    }

                }

            }

        }

    }

    return cr_make_param_array(struct glob_merg_step_kern_param_tupl, test_params, num_tups, free_glob_merg_step_kern_params);

}

ParameterizedTest(struct glob_merg_step_kern_param_tupl *test_tupl, params, glob_merg_step_kern_tests,
                   .init = setup_glob_merg_step_kern_tests, .fini = teardown_glob_merg_step_kern_tests) {

}



