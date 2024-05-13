
#include <stdbool.h>
#include "iterative_cuda_mergesort_ytl.h"
extern "C" {
    #include "main.h"
}

// Number of merge phases each threadblock executes
#define MAX_NUM_MERGE_PHASES_PER_TBLOCK 256
// Think twice before changing any macro definition below this comment.
#define NUM_MS_PER_SEC 1000.0f
#define NUM_TOTAL_ARRAYS_MERGESORT NUM_ARRAYS_PER_MERGE
#define NUM_INDICIES_PER_SUBARRAY 2


/*!
 * Given the first five parameters, compute and store within the second 5 parameters
 * what they ought to be; eventual goal is to merge the two array groups/subarrays
 * stored in aux_arr into data_arr.
 *
 * \param data_arr - the pointer to beginning of array into which to store the
 *                   final merged results.
 * \param aux_arr - the pointer to beginning of array from which to acquire the
 *                  array elements to be merged.
 * \param left_start_aux_arr - beginning of relevant "left portion" of aux_arr;
 *                             this beginning marks the start of where to retrieve
 *                             the first group of relevant array elements to be merged.
 * \param mid_aux_arr - end of "left portion" of aux_arr and beginning of "right portion"
 *                      of aux_arr; marks the start of where to retrieve the second
 *                      group of relevant array elements to be merged.
 * \param right_aux_arr - end of "right portion" aux_arr; marks the end of where to retrieve
 *                        second array group/subarray.
 * 
 * \param first_input_arr - where the pointer to the beginning of the "left portion" of
 *                          aux_arr gets stored
 * \param second_input_arr - where the pointer to the beginning of the "right portion" of
 *                           aux_arr gets stored
 * \param dest_arr - this is the pointer to the beginning of the portion of data_arr into
 *                   which to store the merged results of the "left" and "right" portions
 *                   of aux_arr
 * \param first_input_arr_len - this is the pointer into which to store the length of the
 *                              "left portion" of the aux_arr
 * \param second_input_arr_len - this is the pointer into which to store the length of the
 *                              "right portion" of the aux_arr
 *
 */
__device__  void get_basic_merge_params(double *data_arr,
                                                        double *aux_arr,
                                                        long left_start_aux_arr,
                                                        long mid_aux_arr,
                                                        long right_aux_arr,
                                                        double **first_input_arr,
                                                        double **second_input_arr,
                                                        double **dest_arr,
                                                        long *first_input_arr_len,
                                                        long *second_input_arr_len) {

    *first_input_arr_len = mid_aux_arr - left_start_aux_arr + 1;
    *second_input_arr_len = right_aux_arr - mid_aux_arr;
    *first_input_arr = &aux_arr[left_start_aux_arr];
    if (mid_aux_arr < right_aux_arr) {
        *second_input_arr = &aux_arr[mid_aux_arr + 1];
    } else {
        *second_input_arr = &aux_arr[mid_aux_arr];
    }
    *dest_arr = &data_arr[left_start_aux_arr];

}

/*!
 * Very basic merge function merging two sorted arrays into a destination array
 *  Parameters not fully documented here as they should be self-explanatory.
 */
__device__ void merge(double *first_input_array,
                                      double *second_input_array,
                                      double *dest_array,
                                      long first_input_arr_len,
                                      long second_input_arr_len,
                                      bool sort_non_descending) {

    long current_index_first_arr = 0;
    long current_index_second_arr = 0;
    long current_index_dest_arr = 0;

    /*!
     * While the iterators to first and second input arrays
     * have both not exceeded their bounds, and if the two elements
     * that the iterators point to are of different values, simply
     * copy over the smaller (OR larger if sort non-ascending) element
     * to the destination array and then increment both the iterator of
     * the associated origin array and the iterator of the destination
     * array. If the elements are of equal values, copy from the first
     * input array to ensure a stable merge before incrementing the
     * appropriate iterators.
     */
    while (current_index_first_arr < first_input_arr_len &&
               current_index_second_arr < second_input_arr_len) {

      if (sort_non_descending ?
               (first_input_array[current_index_first_arr] <=
                    second_input_array[current_index_second_arr]) :
               (first_input_array[current_index_first_arr] >=
                         second_input_array[current_index_second_arr])) {
           dest_array[current_index_dest_arr++] =
                 first_input_array[current_index_first_arr++];
      } else {
           dest_array[current_index_dest_arr++] =
                 second_input_array[current_index_second_arr++];
      }

    }

    /*!
     * Dump rest of contents of either the first or second
     * input array into the destination array depending on whichever
     * one still has elements remaining to be merged.
     */
    if (current_index_first_arr >= first_input_arr_len) {

        for (; current_index_second_arr < second_input_arr_len;
                                       ++current_index_second_arr) {
            dest_array[current_index_dest_arr++] =
                        second_input_array[current_index_second_arr];
        }

    } else {

        for (; current_index_first_arr < first_input_arr_len;
                                        ++current_index_first_arr) {
            dest_array[current_index_dest_arr++] =
                       first_input_array[current_index_first_arr];
        }

    }

}


/*!
 * This function returns the index of the co-rank of the array
 * represented by first_input_arr, based on the starting index
 * of the destination array. Copies of contents from each of the
 * two input arrays (first_input_arr and second_input_arr) will
 * be merged into the destination array using ordered merging
 * starting at the aforementioned destination array starting index.
 * The co-rank returned is a zero-based index which represents how
 * much of the array that first_input_arr points to shall be excluded
 * from the aforementioned ordered merging. The behavior of this function
 * is undefined if dest_arr_starting_index is greater than the sum of
 * first_input_arr_len and second_input_arr_len, and if first_input_arr
 * and second_input_arr each point to unsorted arrays or arrays which are
 * sorted in the order that is opposite to the order denoted by the
 * sort_non_descending parameter
 *
 * \param dest_arr_starting_index - starting index of destination array
 * \param first_input_arr - pointer to beginning of first input array
 * \param first_input_arr_len - total length of first input array
 * \param second_input_arr - pointer to beginning of second input array
 * \param second_input_arr_len - total length of second input array
 * \param sort_non_descending - the order in which the merging(s) will occur
 * \return the index of the co-rank of array represented by first_input_arr
 *
 */
__device__ long determine_corank(long dest_arr_starting_index,
                                                 double *first_input_arr,
                                                 long first_input_arr_len,
                                                 double *second_input_arr,
                                                 long second_input_arr_len,
                                                 bool sort_non_descending) {
    /*!
     * Smallest and largest possible values of the co-rank to be returned.
     */
    long corank_lower_bound, corank_upper_bound;
    /*!
     * If the starting index in destination array is large enough that
     * it implies that at least some portion of the array referenced by
     * first_input_arr must be a part of the destination array at indices
     * less than the aforementioned starting index, then initialize lower
     * bound to include the that minimum portion; otherwise the lower bound
     * must be zero.
     */
    if (dest_arr_starting_index > second_input_arr_len) {
       corank_lower_bound = dest_arr_starting_index - second_input_arr_len;
    } else {
       corank_lower_bound = 0;
    }
    /*!
     * The co-rank upper bound must be set to include some portion or all of
     * the first input array's contents, depending on how large the destination
     * array starting index is; only when the destination array starting index
     * is exactly zero will the aforementioned portion be nothing.
     */
    if (dest_arr_starting_index < first_input_arr_len) {
        corank_upper_bound = dest_arr_starting_index;
    } else {
        corank_upper_bound = first_input_arr_len;
    }

    /*!
     * The co-rank search MUST stop when the largest and smallest possible
     * values of the co-rank converge.
     */
    while (corank_lower_bound < corank_upper_bound) {

        /*!
         * Guess for co-rank of array referenced by first_input_arr; initialize
         * to the average of the upper and lower bounds as we are searching
         * for the co-rank in a binary search fashion, which is possible given
         * that this function assumes that both input arrays are already sorted
         * and the ordering used for the sorting is the same for both of them.
         */
        long current_corank_first_arr = corank_lower_bound +
                                         (corank_upper_bound -
                                              corank_lower_bound) / 2;
        /*!
         * Compute what the corresponding co-rank of the array referenced by
         * second_input_arr would be given the current guess of co-rank of array
         * referenced by first_input_arr.
         */
        long current_corank_second_arr =
                dest_arr_starting_index - current_corank_first_arr;

        /*
         * Will the co-rank guess cause there to be too much
         * elements merged from the second array?
         */
        bool over_included_second_array;
        /*
         * Will the co-rank guess cause there to be too much
         * elements merged from the first array?
         */
        bool over_included_first_array;

        /*
         * Following if statement guarantees that memory accesses
         * needed to do element comparisons are grouped together logically;
         * please refer to the next two comment blocks for what
         * the comparisons in this if-else statement mean.
         */
        if (sort_non_descending) {

            over_included_second_array =
                   (first_input_arr[current_corank_first_arr - 1] >
                              second_input_arr[current_corank_second_arr]);
            over_included_first_array =
                   (first_input_arr[current_corank_first_arr] <=
                              second_input_arr[current_corank_second_arr - 1]);

        } else {

            over_included_second_array =
                   (first_input_arr[current_corank_first_arr - 1] <
                              second_input_arr[current_corank_second_arr]);
            over_included_first_array =
                   (first_input_arr[current_corank_first_arr] >=
                              second_input_arr[current_corank_second_arr - 1]);

        }

        /*!
         * If the current guess for the co-rank of array referenced by
         * first_input_arr has not been lowered past the beginning of
         * the first input array, the guess of the co-rank of array
         * referenced by second_input_arr has not been increased past the end of
         * the array referenced by second_input_arr based on the current guess
         * for the co-rank of the first input array, AND if the current
         * guess of the co-rank of the first input array is too high (i.e. the co-rank
         * references an index in the first input array which when used to compute
         * the co-rank for the second input array, would result in the inclusion of
         * elements from the second input array which will be less than [OR greater than
         * if sort non-ascending] the element at index "dest_arr_starting_index - 1" in
         * the destination array), then lower the upper bound of the guess of the co-rank
         * of first input array by excluding any co-rank candidates for the first input
         * array that is greater than or equal to the current co-rank guess for the first
         * input array.
         */
        if (current_corank_first_arr > 0 &&
                current_corank_second_arr < second_input_arr_len &&
                                            over_included_second_array) {

            corank_upper_bound = current_corank_first_arr - 1;

        /*!
         * If the current guess for the co-rank of array referenced by
         * second_input_arr has not been lowered past the beginning of
         * the second input array based on the current guess for the
         * co-rank of the first input array, the current guess of co-rank of
         * array referenced by first_input_arr has not been increased past
         * the end of the array referenced by first_input_arr, AND if the
         * current guess of the co-rank of first input array is too low (i.e.
         * the co-rank references an index in the first input array which would
         * result in the inclusion of elements from the first input array which
         * will be less than OR EQUAL TO [OR greater than OR EQUAL TO if sort
         * non-ascending] the element at index "des_arr_starting_index -
         * 1" in the destination array), then increase the lower bound of the
         * guess of the co-rank of first input array by excluding any co-rank
         * candidates for the first input array that is less than or equal to
         * the current co-rank guess for the first input array. The equality portion
         * of the comparison is necessary to ensure all merges that depend on this
         * "determine_corank" function are stable.
         */
        } else if (current_corank_second_arr > 0 &&
                       current_corank_first_arr < first_input_arr_len &&
                                                 over_included_first_array) {

            corank_lower_bound = current_corank_first_arr + 1;

        /*!
         * Return the co-rank guess for the first input array if it does not
         * need any more adjustment based on the immediate previous two
         * if statements.
         */
        } else {

            return current_corank_first_arr;

        }

    }

    /*!
     * If the upper and lower co-rank bounds converge when attempting to find
     * the co-rank, then just return the lower bound.
     */
    return corank_lower_bound;

}


/*!
 * Using the determine_corank device function, load-balances/parallelizes the
 * process of merging two sorted input arrays into an output array across
 * one threadblock ONLY and then proceeds to do the actual merging using the
 * standard simple sequential merge algorithm after each thread has been assigned
 * the appropriate amount of work. This function is used as an intermediate sorting
 * stage for an overall mergesort algorithm where the two input arrays and the output
 * array being worked on all are small enough to fit in CUDA shared memory/HIP local
 * data store/etc. but it'll be too inefficient to assign only one thread to the
 * merging of two input array pairs.
 *
 * For the following function to actually merge the input arrays into the output
 * array properly, the following conditions must be met for the program to not
 * crash or otherwise go into "undefined behavior territory":
 * 1. num_threads_per_merge must be greater than zero
 * 2. num_threads_per_merge must be less than or equal to number of threads
 *    launched per threadblock
 * 3. The caller must ensure that the memory region accessed by each threadblock
 *    does not overlap with memory regions accessed by other threadblocks
 *
 * \param first_input_array pointer to beginning of the first input array to be
 *                          merged into the output array.
 * \param second_input_array pointer to beginning of the second input array to be
 *                          merged into the output array.
 * \param dest_array pointer to beginning of output array
 * \param first_input_array_len length of array pointed to by first_input_array
 * \param second_input_array_len length of array pointed to by second_input_array
 * \param num_threads_per_merge number of threads within a threadblock that'll be used
 *                              to load balance a merge that would otherwise happen
 *                              sequentially.
 * \param sort_non_descending - the order in which the merging(s) will occur
 */
__device__ void block_level_parallel_merge_using_corank(double *first_input_array,
                                                                        double *second_input_array,
                                                                        double *dest_array,
                                                                        long first_input_array_len,
                                                                        long second_input_array_len,
                                                                        long num_threads_per_merge,
                                                                        bool sort_non_descending) {

       /*!
        * Length of portion of output array that each group of threads within
        * this threadblock is responsible for; each group is used
        * to load balance a merge that would otherwise happen sequentially.
        */
       long dest_arr_chunk_size = 1 + ((first_input_array_len +
                                           second_input_array_len - 1) /
                                                     num_threads_per_merge);

       /*!
        * Starting index of output array for merging process
        * for each group of threads.
        */
       long dest_arr_starting_index =
                  min(first_input_array_len + second_input_array_len,
                             (threadIdx.x % num_threads_per_merge) * dest_arr_chunk_size);

       /*! Ending index of output array for each group of threads */
       long dest_arr_ending_index =
                  min(first_input_array_len + second_input_array_len,
                                           dest_arr_starting_index + dest_arr_chunk_size);

       /*!
        * Index of where the merging process will begin in the first input
        * array for each thread.
        */
       long first_arr_starting_index =
                 determine_corank(dest_arr_starting_index,
                                    first_input_array, first_input_array_len,
                                        second_input_array, second_input_array_len,
                                                                   sort_non_descending);
       /*!
        * Index of where the merging process will end in the first input
        * array for each thread.
        */
       long first_arr_ending_index =
                 determine_corank(dest_arr_ending_index,
                                     first_input_array, first_input_array_len,
                                        second_input_array, second_input_array_len,
                                                                  sort_non_descending);

       /*! Where merging process starts in second input array. */
       long second_arr_starting_index =
                 dest_arr_starting_index - first_arr_starting_index;
       /*! Where merging process ends in second input array. */
       long second_arr_ending_index =
                 dest_arr_ending_index - first_arr_ending_index;

       // Do actual load-balanced merges
       merge(&first_input_array[first_arr_starting_index],
                  &second_input_array[second_arr_starting_index],
                      &dest_array[dest_arr_starting_index],
                         first_arr_ending_index - first_arr_starting_index,
                             second_arr_ending_index - second_arr_starting_index,
                                                                sort_non_descending);


}

#ifndef CIRC_BUFF

/*!
 * Using the determine_corank device function, load-balances/parallelizes the
 * process of merging two sorted input arrays into an output array across
 * MULTIPLE threadblocks and then proceeds to do the actual merging using the
 * standard simple sequential merge algorithm after each thread has been assigned
 * the appropriate amount of work AND the appropriate amount of elements from each
 * input array has been cached within linear caching buffers. This function is
 * used as an intermediate sorting stage for an overall mergesort algorithm where
 * the two input arrays and the output array being worked on all are TOO LARGE to
 * fit in CUDA shared memory/HIP local data store/etc. but it'll be too inefficient
 * to use EXCLUSIVELY CUDA/HIP global memory in order to do each merging process.
 * This implementation of merging parallelization uses shared memory/local data
 * store as a linear caching buffer.
 *
 * For the following function to actually merge the input arrays into the output
 * array properly, the following conditions must be met for the program to not
 * crash or otherwise go into "undefined behavior territory":
 * 1. data_buff_len_per_arr must be greater than zero
 * 2. max_num_blocks_per_virtual_grid must be greater than zero
 * 3. num_blocks_this_virtual_grid must be greater than zero AND less than or
 *    equal to max_num_blocks_per_virtual_grid
 *
 * \param first_input_array pointer to beginning of the first input array to be
 *                          merged into the output array.
 * \param second_input_array pointer to beginning of the second input array to be
 *                          merged into the output array.
 * \param dest_array pointer to beginning of output array
 * \param first_input_array_len length of array pointed to by first_input_array
 * \param second_input_array_len length of array pointed to by second_input_array
 * \param first_array_buff pointer to beginning of the region of shared memory to
 *                         be used as linear caching buffer for the first input array
 * \param second_array_buff pointer to beginning of the region of shared memory to
 *                         be used as linear caching buffer for the second input array
 * \param data_buff_len_per_arr length of each linear caching buffer for each input array
 * \param max_num_blocks_per_virtual_grid Highest number of participating threadblocks
 *                                        for any given "virtual grid", where each
 *                                        "virtual grid" is a collection of threadblocks
 *                                        which are trying to perform the merging of the
 *                                        same two input arrays from global memory.
 * \param num_blocks_this_virtual_grid Number of participating threadblocks in this
 *                                     "virtual grid"
 * \param sort_non_descending - the order in which the merging(s) will occur
 *
 */
__device__ void global_parallel_merge_using_corank(double *first_input_array,
                                                                   double *second_input_array,
                                                                   double *dest_array,
                                                                   long first_input_array_len,
                                                                   long second_input_array_len,
                                                                   double *first_array_buff,
                                                                   double *second_array_buff,
                                                                   long *corank_broadcast_buff,
                                                                   long data_buff_len_per_arr,
                                                                   long max_num_blocks_per_virtual_grid,
                                                                        long num_blocks_this_virtual_grid,
                                                                                   bool sort_non_descending) {

       /*! 
        *  Compute how much of the output array this
        *  collection of threadblocks will be responsible for.
        */
       long dest_arr_len_per_virtual_grid = 1 + ((first_input_array_len +
                                                    second_input_array_len - 1) /
                                                         num_blocks_this_virtual_grid);
       /*!
        * Compute starting index at which output will be fed into output array
        * given the maximum number of threadblocks per virtual grid (there may
        * be a virtual grid with less than the max number if things don't "divide
        * out evenly" amongst the threadblocks)
        */
       long dest_arr_tblock_start =
                min((blockIdx.x % max_num_blocks_per_virtual_grid) *
                                          dest_arr_len_per_virtual_grid,
                                                    first_input_array_len +
                                                         second_input_array_len);
 
       /*!
        * Compute ending index at which output will be finished feeding into output
        * array given the maximum number of threadblocks per virtual grid.
        */
       long dest_arr_tblock_end =
                min(((blockIdx.x % max_num_blocks_per_virtual_grid) + 1) *
                                                dest_arr_len_per_virtual_grid,
                                    first_input_array_len + second_input_array_len);

       /*!
        * if _this thread_ is the first thread of each threadblock, go ahead and use
        * the determine_corank function to directly access global memory and figure
        * out which portion of the first input array this threadblock will be responsible
        * for given the previous computed values.
        */
       if (threadIdx.x == 0) {
            corank_broadcast_buff[0] = determine_corank(dest_arr_tblock_start,
                                                        first_input_array,
                                                        first_input_array_len,
                                                        second_input_array,
                                                        second_input_array_len,
                                                             sort_non_descending);
            corank_broadcast_buff[1] = determine_corank(dest_arr_tblock_end,
                                                        first_input_array,
                                                        first_input_array_len,
                                                        second_input_array,
                                                        second_input_array_len,
                                                             sort_non_descending);
       }

       // Wait until first thread of each threadblock finishes its work.
       __syncthreads();

       /*!
        * Store within each thread's registers the beginning
        * of "working" portion of first input array
        */
       long first_arr_tblock_start = corank_broadcast_buff[0];
       /*!
        * Store within each thread's registers the end
        * of "working" portion of first input array
        */
       long first_arr_tblock_end = corank_broadcast_buff[1];

       // Compute for second input array start of where input shall be read
       long second_arr_tblock_start = dest_arr_tblock_start -
                                          first_arr_tblock_start;
       // Compute for second input array end of where input shall be read
       long second_arr_tblock_end = dest_arr_tblock_end -
                                          first_arr_tblock_end;

       // Make sure all threads have updated info for threadblock-based input bounds
       __syncthreads();

       // Length of portion of first input array this threadblock is responsible for
       long first_arr_len_per_tblock = first_arr_tblock_end -
                                          first_arr_tblock_start;
       // Length of portion of second input array this threadblock is responsible for
       long second_arr_len_per_tblock = second_arr_tblock_end -
                                          second_arr_tblock_start;
       // Length of portion of output array this threadblock is responsible for
       long dest_arr_len_per_tblock = dest_arr_tblock_end -
                                          dest_arr_tblock_start;

       /*! 
        * How many "round trips" will be made into global memory to fill all
        * of the portion of output array this threadblock is responsible for
        * given the length of each linear caching buffer corresponding to each
        * input array.
        */
       long num_merge_phases = 1 + ((dest_arr_len_per_tblock - 1) / data_buff_len_per_arr);

       // How much of first input array has been transferred to the linear caching buffer
       long first_arr_consumed = 0;
       // How much of second input array has been transferred to the linear caching buffer
       long second_arr_consumed = 0;
       // How much of merging into the output/destination array has been done
       long dest_arr_completed = 0;

       /*!
        * Iterate over all "round trips" to global memory; each "round trip" to global
        * memory involves first filling up each linear caching buffer with elements not
        * yet transferred from global memory, then computing where to dump the elements
        * from the caching buffers into the destination/output array in global memory,
        * and then directly dumping/merging the contents of the buffers into the destination/
        * output array; ONLY HALF OF THE TOTAL CAPACITY OF BOTH BUFFERS ARE FLUSHED PER
        * ROUND TRIP TO ENSURE CORRECT MERGES.
        */
       for (long merge_phase = 0; merge_phase < num_merge_phases; ++merge_phase) {

                /*
                 * Do textbook coalesced global memory accesses via having each thread
                 * loop over its appropriate set of valid indicies for both linear
                 * caching buffers. For example, if threadIdx.x here is 1, that thread
                 * shall be responsible for indicies 1, 257, 513, 769, etc. within each
                 * linear caching buffer IF AND ONLY IF blockDim.x is 256
                 */
                for (long first_thread_pos_in_buff = 0;
                            first_thread_pos_in_buff < data_buff_len_per_arr;
                                   first_thread_pos_in_buff += blockDim.x) {

                     /*
                      * Check to make sure that this thread is not attempting to read more than what is
                      * available in global memory for first input array.
                      */
                     if (first_thread_pos_in_buff + threadIdx.x < first_arr_len_per_tblock - first_arr_consumed) {

                          // Fill the linear caching buffer for the first input array
                          first_array_buff[threadIdx.x + first_thread_pos_in_buff] =
                                first_input_array[first_arr_tblock_start + first_arr_consumed +
                                                                     threadIdx.x + first_thread_pos_in_buff];

                     }

                     /*
                      * Check to make sure that this thread is not attempting to read more than what is
                      * available in global memory for second input array.
                      */
                     if (first_thread_pos_in_buff + threadIdx.x < second_arr_len_per_tblock - second_arr_consumed) {

                          // Fill the linear caching buffer for the second input array
                          second_array_buff[threadIdx.x + first_thread_pos_in_buff] =
                                second_input_array[second_arr_tblock_start + second_arr_consumed +
                                                                     threadIdx.x + first_thread_pos_in_buff];

                     }


               }

               // Wait until all linear caching buffers have been completely filled
               __syncthreads();

               /*!
                * Compute where each thread shall start dumping/merging cached elements from input
                * arrays into the output/destination array, where the index computed is an offset based off
                * of the portion of the output array that this threadblock is responsible for as well
                * as how much of the output array this threadblock has completed filling.
                */
               long dest_arr_thread_start = min(threadIdx.x * (data_buff_len_per_arr / blockDim.x),
                                                    dest_arr_len_per_tblock - dest_arr_completed);
               /*!
                * Compute where each thread shall end its dumping/merging of cached elements from input
                * arrays; the offset corresponding to this index is exactly the same as the offset for
                * the index "dest_arr_thread_start".
                */
               long dest_arr_thread_end = min((threadIdx.x + 1) * (data_buff_len_per_arr / blockDim.x),
                                                    dest_arr_len_per_tblock - dest_arr_completed);

               /*!
                * Determine where the merging shall start for each thread for the linear caching buffer
                * corresponding to the first input array; 0 for this index means start of the linear caching
                * buffer, NOT THE ACTUAL PORTION OF THE INPUT ARRAY IN GLOBAL MEMORY WHICH GOT CACHED.
                */
               long first_arr_thread_start = determine_corank(dest_arr_thread_start,
                                                              first_array_buff,
                                                              min(data_buff_len_per_arr,
                                                                      first_arr_len_per_tblock -
                                                                                  first_arr_consumed),
                                                              second_array_buff,
                                                              min(data_buff_len_per_arr,
                                                                      second_arr_len_per_tblock -
                                                                                  second_arr_consumed),
                                                                                      sort_non_descending);
               /*!
                * Determine where the merging shall end for each thread for the linear caching buffer
                * corresponding to the first input array; 0 for this index AGAIN means start of the
                * linear caching buffer.
                */
                long first_arr_thread_end = determine_corank(dest_arr_thread_end,
                                                            first_array_buff,
                                                            min(data_buff_len_per_arr,
                                                                      first_arr_len_per_tblock -
                                                                                 first_arr_consumed),
                                                            second_array_buff,
                                                            min(data_buff_len_per_arr,
                                                                    second_arr_len_per_tblock -
                                                                                 second_arr_consumed),
                                                                                     sort_non_descending);

               // Where merging shall start within the second input array's linear caching buffer
               long second_arr_thread_start = dest_arr_thread_start - first_arr_thread_start;
               // Where merging shall end within the second input array's linear caching buffer
               long second_arr_thread_end = dest_arr_thread_end - first_arr_thread_end;

               // Do load-balanced merges from caching buffers into output/destination array in global memory
               merge(&first_array_buff[first_arr_thread_start], &second_array_buff[second_arr_thread_start],
                     &dest_array[dest_arr_tblock_start + dest_arr_completed + dest_arr_thread_start],
                     first_arr_thread_end - first_arr_thread_start,
                                        second_arr_thread_end - second_arr_thread_start, sort_non_descending);

               /*!
                * Since only at most half of the total capacity of both caching buffers are flushed to
                * the output/destination array per "round trip" to global memory, figure out how much
                * of the caching buffer for the first input array has been flushed.
                */
               first_arr_consumed +=
                     determine_corank(min(data_buff_len_per_arr,
                                             dest_arr_len_per_tblock - dest_arr_completed),
                                        first_array_buff,
                                        min(data_buff_len_per_arr,
                                             first_arr_len_per_tblock - first_arr_consumed),
                                        second_array_buff,
                                        min(data_buff_len_per_arr,
                                             second_arr_len_per_tblock - second_arr_consumed),
                                                                             sort_non_descending);
               /*!
                * Increment how much has been completed for destination/output array by
                * how much has been flushed to it.
                */
               dest_arr_completed += min(data_buff_len_per_arr, dest_arr_len_per_tblock - dest_arr_completed);
               // Compute how much of second input array in global memory has been consumed.
               second_arr_consumed = dest_arr_completed - first_arr_consumed;

               // Wait till all merges are done before cycling to next global memory "round trip".
               __syncthreads();

       }


}

#endif

/*!
 * This is a global kernel function which uses CUDA shared memory/HIP local data
 * store/etc. as a cache to create contiguous sorted sub-arrays, each of length
 * buff_size. This kernel function essentially uses the standard mergesort load-
 * balanced over all CUDA threads launched in order to create the sorted sub-arrays.
 * This function covers the first two stages for an overall mergesort algorithm, and
 * the two different stages have been combined into one as to not create functions
 * which are too fragmented across the entire program.  The first stage is sorting
 * by individual CUDA threads, and the second stage is cooperative sorting by sub-
 * groups of CUDA threads within a CUDA threadblock.
 * 
 * Following conditions must be met for function to behave properly:
 * 1. input_array and output_array each must point to regions which have already
 *    have memory allocated (i.e. NOT pointing to NULL or free'd memory regions)
 * 2. array_len MUST be greater than 0
 * 3. buff_size MUST be greater than 0
 *
 * \param input_array pointer to beginning of array in global memory to have its
 *                    elements sorted into contiguous subarrays each of length buff_size
 * \param output_array pointer to beginning of array in global memory to store
 *                     the resulting contiguously sorted subarrays
 * \param array_len the length of the array represented by input_array, which should
 *                  be the same as the length of the array represented by output_array.
 * \param buff_size The amount of shared memory to be used as a cache to be allocated
 *                  to each threadblock, this is also the length of each contiguous
 *                  sub-array to be created from the input_array.
 * \param sort_non_descending whether or not to sort the "input array" in a non descending
 *                            manner.
 */
__global__ void shared_mem_mergesort_subarrays(double *input_array,
                                                double *output_array,
                                                long array_len,
                                                long buff_size,
                                                bool sort_non_descending) {


       extern __shared__ double array_and_array_aux_buff[];
       __shared__ double *buffer_ptrs[NUM_TOTAL_ARRAYS_MERGESORT];
       double *arr_buff = &array_and_array_aux_buff[0];
       double *arr_aux_buff = &array_and_array_aux_buff[buff_size];
       long last_tblock_buff_len = array_len % buff_size;

       // Load subarrays into shared auxillary_buffer
       for (long first_thread_pos_buff = 0;
              first_thread_pos_buff < buff_size;
                 first_thread_pos_buff += blockDim.x) {

            if (blockIdx.x * buff_size +
                   first_thread_pos_buff + threadIdx.x < array_len) {

                arr_buff[first_thread_pos_buff + threadIdx.x] =
                               input_array[blockIdx.x * buff_size +
                                     first_thread_pos_buff + threadIdx.x];

            }

       }

       __syncthreads();


       // First stage of kernel - per thread sorting
       long per_block_arr_len = buff_size;
       long per_thread_max_len = buff_size / blockDim.x;
       long per_thread_arr_len = per_thread_max_len;

       if ((blockIdx.x + 1) * buff_size > array_len) {

           per_block_arr_len = last_tblock_buff_len;

       }

       if ((threadIdx.x + 1) * per_thread_max_len > per_block_arr_len) {

           if (threadIdx.x * per_thread_max_len >= per_block_arr_len) {

               per_thread_arr_len = 0;

           } else {

               per_thread_arr_len = per_block_arr_len - (threadIdx.x * per_thread_max_len);

           }

       }

       for (long current_sorted_size = 1;
                     current_sorted_size <= per_thread_arr_len - 1;
                                         current_sorted_size *= NUM_ARRAYS_PER_MERGE) {

           // Swap arr_aux_buff and arr_buff pointers to avoid unneccessary copying
           double *temp_buff_ptr = arr_aux_buff;
           arr_aux_buff = arr_buff;
           arr_buff = temp_buff_ptr;

           for (long thread_left_start_buff = threadIdx.x * per_thread_max_len;
                     thread_left_start_buff <
                          threadIdx.x * per_thread_max_len + per_thread_arr_len;
                                   thread_left_start_buff += (NUM_ARRAYS_PER_MERGE *
                                                                     current_sorted_size)) {

               long thread_mid_in_buff =
                       min(thread_left_start_buff + current_sorted_size - 1,
                               threadIdx.x * per_thread_max_len + per_thread_arr_len - 1);
               long thread_right_in_buff =
                       min(thread_left_start_buff +
                              NUM_ARRAYS_PER_MERGE * current_sorted_size - 1,
                                 threadIdx.x * per_thread_max_len + per_thread_arr_len - 1);

               long arr_left_len = 0, arr_right_len = 0;
               double *left_array_beginning = NULL, *right_array_beginning = NULL,
                                                         *dest_array_beginning = NULL;
               get_basic_merge_params(arr_buff, arr_aux_buff, thread_left_start_buff,
                                           thread_mid_in_buff, thread_right_in_buff,
                                           &left_array_beginning, &right_array_beginning,
                                           &dest_array_beginning, &arr_left_len, &arr_right_len);

               merge(left_array_beginning, right_array_beginning, dest_array_beginning,
                                            arr_left_len, arr_right_len, sort_non_descending);

           }

       }


       // Sync up buffers
       if (threadIdx.x == 0) {

           buffer_ptrs[0] = arr_buff;
           buffer_ptrs[1] = arr_aux_buff;

       }

       __syncthreads();

       double *first_thread_arr_buff_ptr = buffer_ptrs[0];
       double *first_thread_arr_aux_buff_ptr = buffer_ptrs[1];

       // Handling edge case for buffer pointers swapping
       if (first_thread_arr_aux_buff_ptr == arr_buff &&
               first_thread_arr_buff_ptr == arr_aux_buff) {

           if (per_thread_arr_len > 0) {

               for (long thread_offset = threadIdx.x * per_thread_max_len;
                           thread_offset <
                                threadIdx.x * per_thread_max_len + per_block_arr_len;
                                                                        ++thread_offset) {

                   arr_aux_buff[thread_offset] = arr_buff[thread_offset];

               }

           }

           arr_buff = first_thread_arr_buff_ptr;
           arr_aux_buff = first_thread_arr_aux_buff_ptr;

       }

       long num_threads_per_merge = NUM_ARRAYS_PER_MERGE;

       __syncthreads();


       // second stage of kernel - intra-block sorting
       for (long current_sorted_size = buff_size / blockDim.x;
                        current_sorted_size <= per_block_arr_len - 1;
                                       current_sorted_size *= NUM_ARRAYS_PER_MERGE) {

           // Swap arr_aux_buff and arr_buff pointers to avoid unneccessary copying
           double *temp_buff_ptr = arr_aux_buff;
           arr_aux_buff = arr_buff;
           arr_buff = temp_buff_ptr;

           __syncthreads();

           long thread_group_left_start_in_buff =
                     (threadIdx.x / num_threads_per_merge) *
                              current_sorted_size * NUM_ARRAYS_PER_MERGE;

           if (thread_group_left_start_in_buff < per_block_arr_len) {

               long thread_group_mid_in_buff =
                      min(thread_group_left_start_in_buff + current_sorted_size - 1,
                                                                 per_block_arr_len - 1);
               long thread_group_right_in_buff =
                      min(thread_group_left_start_in_buff +
                                NUM_ARRAYS_PER_MERGE * current_sorted_size - 1,
                                                                 per_block_arr_len - 1);

               long arr_left_len = 0, arr_right_len = 0;
               double *left_array_beginning = NULL, *right_array_beginning = NULL,
                                                       *dest_array_beginning = NULL;
               get_basic_merge_params(arr_buff, arr_aux_buff, thread_group_left_start_in_buff,
                                           thread_group_mid_in_buff, thread_group_right_in_buff,
                                           &left_array_beginning, &right_array_beginning,
                                           &dest_array_beginning, &arr_left_len, &arr_right_len);

               block_level_parallel_merge_using_corank(left_array_beginning, right_array_beginning,
                                                         dest_array_beginning, arr_left_len, arr_right_len,
                                                         num_threads_per_merge, sort_non_descending);

           }

           num_threads_per_merge *= NUM_ARRAYS_PER_MERGE;

           __syncthreads();

       }


       // Load sorted buffers into global memory
       for (long first_thread_pos_buff = 0;
              first_thread_pos_buff < buff_size;
                 first_thread_pos_buff += blockDim.x) {

            if (blockIdx.x  *buff_size +
                   first_thread_pos_buff + threadIdx.x < array_len) {


                output_array[blockIdx.x * buff_size +
                               first_thread_pos_buff + threadIdx.x] =
                                  arr_buff[first_thread_pos_buff + threadIdx.x];


            }

       }


}

/*!
 * This is a global kernel function which serves as a single step in doing each
 * of the series of merges in a standard "mergesort".
 */
__global__ void global_mem_mergesort_step(double *aux_array,
                                          double *array,
                                          long array_len,
                                          long current_sorted_size,
                                          long max_blocks_per_virtual_grid,
                                          long buff_size, bool sort_non_descending) {

       extern __shared__ double first_n_second_aux_subarr_buff[];
       __shared__ long corank_broadcast[NUM_INDICIES_PER_SUBARRAY];
       double *first_aux_subarr_buff = &first_n_second_aux_subarr_buff[0];
       double *second_aux_subarr_buff = &first_n_second_aux_subarr_buff[buff_size];

       long virtual_grid_num = blockIdx.x / max_blocks_per_virtual_grid;
       long num_blocks_this_virtual_grid = max_blocks_per_virtual_grid;
       if ((virtual_grid_num + 1) * max_blocks_per_virtual_grid > gridDim.x) {
           num_blocks_this_virtual_grid = gridDim.x % max_blocks_per_virtual_grid;
       }

       long block_group_left_start = (blockIdx.x / max_blocks_per_virtual_grid) *
                                            current_sorted_size * NUM_ARRAYS_PER_MERGE;
       long block_group_mid = min(block_group_left_start + current_sorted_size - 1,
                                                                        array_len - 1);
       long block_group_right_end = min(block_group_left_start +
                                              NUM_ARRAYS_PER_MERGE * current_sorted_size - 1,
                                                                                  array_len - 1);

       long arr_left_len = 0, arr_right_len = 0;
       double *left_array_beginning = NULL, *right_array_beginning = NULL,
                                               *dest_array_beginning = NULL;
       get_basic_merge_params(array, aux_array, block_group_left_start, block_group_mid,
                               block_group_right_end, &left_array_beginning, &right_array_beginning,
                                                  &dest_array_beginning, &arr_left_len, &arr_right_len);

       global_parallel_merge_using_corank(left_array_beginning, right_array_beginning,
                                           dest_array_beginning, arr_left_len,
                                           arr_right_len, first_aux_subarr_buff,
                                           second_aux_subarr_buff, corank_broadcast,
                                           buff_size, max_blocks_per_virtual_grid,
                                           num_blocks_this_virtual_grid, sort_non_descending);

}

/* Iterative parallel_merge_sort function to sort arr[0...n-1] */
void parallel_merge_sort(double *arr, long arr_len, bool sort_non_descending) {

    double *arr_on_device, *arr_aux_on_device;
    cudaEvent_t start_on_device, stop_on_device;
    float sort_time_without_host_transfers = 0.0f;
    CHECK(cudaEventCreate(&start_on_device));
    CHECK(cudaEventCreate(&stop_on_device));
    CHECK(cudaMalloc(&arr_on_device, arr_len * sizeof(*arr_on_device)));
    CHECK(cudaMalloc(&arr_aux_on_device, arr_len * sizeof(*arr_aux_on_device)));

    CHECK(cudaMemcpy(arr_aux_on_device, arr, arr_len * sizeof(*arr), cudaMemcpyHostToDevice));

    CHECK(cudaEventRecord(start_on_device));

    dim3 block_size_shared_mem_sort(THREADS_PER_BLOCK_SHARED_MEM_SORT, 1, 1);
    dim3 grid_size_shared_mem_sort(1 + ((arr_len - 1) / (TILE_SIZE_SHARED_MEM_SORT)), 1, 1);
    long shared_mem_sort_buff_size = TILE_SIZE_SHARED_MEM_SORT;
    void *shared_mem_sort_kernel_args[] = {&arr_aux_on_device, &arr_on_device,
                                             &arr_len, &shared_mem_sort_buff_size, &sort_non_descending};

    CHECK(cudaLaunchKernel((void*)shared_mem_mergesort_subarrays, grid_size_shared_mem_sort,
                              block_size_shared_mem_sort, shared_mem_sort_kernel_args,
                                 NUM_ARRAYS_PER_MERGE * shared_mem_sort_buff_size * sizeof(*arr_on_device), NULL));

    long init_global_mem_sorted_size = shared_mem_sort_buff_size;
    long global_mem_sort_buff_size_per_arr = TILE_SIZE_GLOBAL_MEM_SORT;
    long num_merge_phases_per_tblock_global_mem_sort =
            (NUM_ARRAYS_PER_MERGE * init_global_mem_sorted_size) / global_mem_sort_buff_size_per_arr;

    for (long current_sorted_size = init_global_mem_sorted_size;
                              current_sorted_size <= arr_len - 1;
                                         current_sorted_size *= NUM_ARRAYS_PER_MERGE) {

         // Swap array pointers to avoid unnecessary copying
         double *temp_array_ptr = arr_aux_on_device;
         arr_aux_on_device = arr_on_device;
         arr_on_device = temp_array_ptr;

         long arr_len_per_block = global_mem_sort_buff_size_per_arr * num_merge_phases_per_tblock_global_mem_sort;
         long max_blocks_per_virtual_grid = (NUM_ARRAYS_PER_MERGE * current_sorted_size) / arr_len_per_block;
         dim3 block_size_global_mem_sort(THREADS_PER_BLOCK_GLOBAL_MEM_SORT, 1, 1);
         dim3 grid_size_global_mem_sort(1 + ((arr_len - 1) / arr_len_per_block), 1, 1);
         void *global_mem_sort_kernel_args[] = {&arr_aux_on_device, &arr_on_device, &arr_len,
                                                 &current_sorted_size, &max_blocks_per_virtual_grid,
                                                 &global_mem_sort_buff_size_per_arr, &sort_non_descending};

         CHECK(cudaLaunchKernel((void*)global_mem_mergesort_step, grid_size_global_mem_sort,
                                   block_size_global_mem_sort, global_mem_sort_kernel_args,
                                   NUM_TILES_GLOBAL_MEM_SORT * global_mem_sort_buff_size_per_arr *
                                                                             sizeof(*arr_on_device), NULL));

         if (num_merge_phases_per_tblock_global_mem_sort < MAX_NUM_MERGE_PHASES_PER_TBLOCK) {
              num_merge_phases_per_tblock_global_mem_sort *= NUM_ARRAYS_PER_MERGE;
         }

    }

    CHECK(cudaEventRecord(stop_on_device));
    CHECK(cudaEventSynchronize(stop_on_device));
    CHECK(cudaEventElapsedTime(&sort_time_without_host_transfers, start_on_device, stop_on_device));

    printf("Time took for mergesort to complete "
               "WITHOUT copy from and to host: %f seconds\n", sort_time_without_host_transfers / NUM_MS_PER_SEC);

    CHECK(cudaMemcpy(arr, arr_on_device, arr_len * sizeof(*arr_on_device), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(arr_on_device));
    CHECK(cudaFree(arr_aux_on_device));
    CHECK(cudaEventDestroy(start_on_device));
    CHECK(cudaEventDestroy(stop_on_device));

}

double u64_to_double_conv(uint64_t in_val) {

    double out_val = 0.0;
    memcpy(&out_val, &in_val, sizeof(out_val));
    return out_val;

}

