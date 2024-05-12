
/*!
 * Unit testing for cuda kernels from main cuda mergesort program
 */

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
static double *test_input_array = NULL;
static double *test_output_array = NULL;
static double *test_input_array_gpu = NULL;
static double *test_output_array_gpu = NULL;
static long sorted_subarray_sizes[] =
         {1, 2, 3, 37,
        257, 512, 8209, 32771,
                        65536};
static long arr_nums_sorted_subarrays[] =
         {1, 2, 3, 37, 257, 1024};
static bool arr_appnd_nxt_smallest_subarr[] = 
         {true, false};
static bool sort_non_desc_arr[] =
         {true, false};

void setup_glob_merg_step_kern_tests(void) {

    set_seed_splitmix64(RAND_NUM_SEED);
    init_xoshiro256starstar();

}

void teardown_glob_merg_step_kern_tests(void) {


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

    struct glob_merg_step_kern_param_tupl *params =
        (struct glob_merg_step_kern_param_tupl*) cr_malloc(sizeof(struct glob_merg_step_kern_param_tupl) * num_tups);


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

                    params[param_idx].sorted_subarray_max_size = sorted_subarray_sizes[index_arr_sorted_subarr_sizes];
                    params[param_idx].total_array_size = sorted_subarray_sizes[index_arr_sorted_subarr_sizes] *
                                                              arr_nums_sorted_subarrays[index_arr_num_sorted_subarrs];
                    params[param_idx].sort_non_descending = sort_non_desc_arr[index_sort_non_desc_arr];

                    if (index_arr_appnd_nxt_smallest_subarr == APPEND_NEXT_SMALLEST_FLAG) {
                        params[param_idx].total_array_size += index_arr_sorted_subarr_sizes <= 0 ?
                                                                0 : sorted_subarray_sizes[index_arr_sorted_subarr_sizes - 1];
                    }


                }

            }

        }

    }

    return cr_make_param_array(struct glob_merg_step_kern_param_tupl, params, num_tups, free_glob_merg_step_kern_params);

}

    
