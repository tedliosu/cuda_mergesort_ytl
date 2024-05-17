
/*!
 * Main header file containing merge sort function and key macros.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// Think twice before changing any macro definition below this comment.
#define NUM_NS_PER_SEC 1000000000.0
#define MAX_USER_INPUT_LEN 21
#define USER_INPUT_NUM_BASE 10
#define NUM_ARRAYS_PER_MERGE 2
#define NUM_TILES_GLOBAL_MEM_SORT NUM_ARRAYS_PER_MERGE

double u64_to_double_conv(uint64_t in_val);
long get_max_arr_len_for_dev(size_t arr_elem_size);
void parallel_merge_sort(double* arr, long arr_len, bool sort_non_descending);
