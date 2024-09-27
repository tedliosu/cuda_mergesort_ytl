
/*!
 * Circular buffer implementation causes program overall to be slower due to over-aggressively
 *    attempting to conserve bandwidth at the expense of reusing existing data cached in L1/L2/etc.
 */

#include "iterative_cuda_mergesort_ytl.h"

#ifdef CIRC_BUFF

__device__ void merge_circular(double* first_input_buff,
                               double* second_input_buff,
                               double* dest_array,
                               long first_input_buff_len,
                               long second_input_buff_len,
                               long first_input_virt_buff_start,
                               long second_input_virt_buff_start,
                                                 long max_buff_len,
                                bool sort_non_descending) {

    long virt_curr_idx_first_buff = 0;
    long virt_curr_idx_second_buff = 0;
    long current_index_dest_arr = 0;

    while (virt_curr_idx_first_buff < first_input_buff_len &&
               virt_curr_idx_second_buff < second_input_buff_len) {

      if (sort_non_descending? (first_input_buff[(first_input_virt_buff_start +
                                                       virt_curr_idx_first_buff) % max_buff_len] <=
              second_input_buff[(second_input_virt_buff_start +
                                         virt_curr_idx_second_buff) % max_buff_len]) : 
              (first_input_buff[(first_input_virt_buff_start +
                                       virt_curr_idx_first_buff) % max_buff_len] >=
              second_input_buff[(second_input_virt_buff_start +
                                         virt_curr_idx_second_buff) % max_buff_len])) {
           dest_array[current_index_dest_arr++] =
                 first_input_buff[(first_input_virt_buff_start +
                                           virt_curr_idx_first_buff) % max_buff_len];
           ++virt_curr_idx_first_buff;
      } else {
           dest_array[current_index_dest_arr++] =
                 second_input_buff[(second_input_virt_buff_start +
                                           virt_curr_idx_second_buff) % max_buff_len];
           ++virt_curr_idx_second_buff;
      }


    }

    if (virt_curr_idx_first_buff >= first_input_buff_len) {

        for (; virt_curr_idx_second_buff < second_input_buff_len;
                                         ++virt_curr_idx_second_buff) {
            dest_array[current_index_dest_arr++] =
                   second_input_buff[(second_input_virt_buff_start +
                                              virt_curr_idx_second_buff) % max_buff_len];
        }

    } else {
        
        for (; virt_curr_idx_first_buff < first_input_buff_len;
                                        ++virt_curr_idx_first_buff) {
            dest_array[current_index_dest_arr++] =
                       first_input_buff[(first_input_virt_buff_start +
                                                virt_curr_idx_first_buff) % max_buff_len];
        }

    }

}

__device__ long determine_virt_corank_circular(long dest_arr_starting_index,
                                              double* first_input_buff,
                                              long first_input_buff_len,
                                              double* second_input_buff,
                                              long second_input_buff_len,
                                              long first_input_buff_start,
                                              long second_input_buff_start,
                                                            long buff_max_len,
                                              bool sort_non_descending) {

    long virt_corank_lower_bound, virt_corank_upper_bound;

    if (dest_arr_starting_index > second_input_buff_len) {
       virt_corank_lower_bound = dest_arr_starting_index - second_input_buff_len;
    } else {
       virt_corank_lower_bound = 0;
    }

    if (dest_arr_starting_index < first_input_buff_len) {
        virt_corank_upper_bound = dest_arr_starting_index;
    } else {
        virt_corank_upper_bound = first_input_buff_len;
    }

    while (virt_corank_lower_bound < virt_corank_upper_bound) {

        long curr_virt_corank_first_arr = virt_corank_lower_bound +
                                           (virt_corank_upper_bound -
                                                virt_corank_lower_bound) / 2;

        long curr_virt_corank_second_arr =
                dest_arr_starting_index - curr_virt_corank_first_arr;

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

        if (sort_non_descending) {

            over_included_second_array =
                  (first_input_buff[(first_input_buff_start +
                                                curr_virt_corank_first_arr - 1) % buff_max_len] >
                              second_input_buff[(second_input_buff_start +
                                                        curr_virt_corank_second_arr) % buff_max_len]);
            over_included_first_array =
                (first_input_buff[(first_input_buff_start +
                                                        curr_virt_corank_first_arr) % buff_max_len] <=
                                    second_input_buff[(second_input_buff_start +
                                                           curr_virt_corank_second_arr - 1) % buff_max_len]);


        } else {

            over_included_second_array =
                  (first_input_buff[(first_input_buff_start +
                                                curr_virt_corank_first_arr - 1) % buff_max_len] <
                              second_input_buff[(second_input_buff_start +
                                                        curr_virt_corank_second_arr) % buff_max_len]);
            over_included_first_array =
                (first_input_buff[(first_input_buff_start +
                                                        curr_virt_corank_first_arr) % buff_max_len] >=
                                    second_input_buff[(second_input_buff_start +
                                                           curr_virt_corank_second_arr - 1) % buff_max_len]);

        }

        if (curr_virt_corank_first_arr > 0 &&
               curr_virt_corank_second_arr < second_input_buff_len &&
                                              over_included_second_array) {
            
            virt_corank_upper_bound = curr_virt_corank_first_arr - 1;

        } else if (curr_virt_corank_second_arr > 0 &&
                        curr_virt_corank_first_arr < first_input_buff_len &&
                                                       over_included_first_array) {

            virt_corank_lower_bound = curr_virt_corank_first_arr + 1;

        } else {

            return curr_virt_corank_first_arr;

        }

    }

    return virt_corank_lower_bound;

}

/*!
 * load balances/parallelizes the merging of two sorted input arrays into an output
 * array from global memory.
 */
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
                                                        bool sort_non_descending) {

       long dest_arr_len_per_virtual_grid = 1 + ((first_input_array_len +
                                                    second_input_array_len - 1) /
                                                         num_blocks_this_virtual_grid);

       long dest_arr_tblock_start =
                min((blockIdx.x % max_num_blocks_per_virtual_grid) *
                                          dest_arr_len_per_virtual_grid,
                                                    first_input_array_len +
                                                         second_input_array_len);

       long dest_arr_tblock_end =
                min(((blockIdx.x % max_num_blocks_per_virtual_grid) + 1) *
                                                dest_arr_len_per_virtual_grid,
                                    first_input_array_len + second_input_array_len);

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

       __syncthreads();

       long first_arr_tblock_start = corank_broadcast_buff[0];
       long first_arr_tblock_end = corank_broadcast_buff[1];

       long second_arr_tblock_start = dest_arr_tblock_start -
                                          first_arr_tblock_start;
       long second_arr_tblock_end = dest_arr_tblock_end -
                                          first_arr_tblock_end;

       __syncthreads();

       long first_arr_len_per_tblock = first_arr_tblock_end -
                                          first_arr_tblock_start;
       long second_arr_len_per_tblock = second_arr_tblock_end -
                                          second_arr_tblock_start;
       long dest_arr_len_per_tblock = dest_arr_tblock_end -
                                          dest_arr_tblock_start;

       long num_merge_phases = 1 + ((dest_arr_len_per_tblock - 1) / data_buff_len_per_arr);

       long first_arr_buff_read_start = 0;
       long second_arr_buff_read_start = 0;
       long first_arr_buff_write_start = first_arr_buff_read_start;
       long second_arr_buff_write_start = second_arr_buff_read_start;
       long first_arr_loaded_total = 0;
       long second_arr_loaded_total = 0;
       long first_arr_processed = 0;
       long second_arr_processed = 0;
       long first_arr_buff_prev_used = data_buff_len_per_arr;
       long second_arr_buff_prev_used = data_buff_len_per_arr;
       long dest_arr_completed = 0;

       for (long merge_phase = 0; merge_phase < num_merge_phases; ++merge_phase) {

               // load first_arr and second_arr into that rediculously small shared mem
               for (long first_thread_pos_in_tile = 0;
                            first_thread_pos_in_tile < first_arr_buff_prev_used;
                                                first_thread_pos_in_tile += blockDim.x) {

                     if (first_thread_pos_in_tile + threadIdx.x <
                              first_arr_len_per_tblock - first_arr_loaded_total &&
                                 first_thread_pos_in_tile + threadIdx.x < first_arr_buff_prev_used) {
                         
                              first_array_buff[(first_arr_buff_write_start +
                                                    threadIdx.x + first_thread_pos_in_tile) %
                                                                             data_buff_len_per_arr] =
                                first_input_array[first_arr_tblock_start + first_arr_loaded_total +
                                                                     threadIdx.x + first_thread_pos_in_tile];

                     }

               }

               for (long first_thread_pos_in_tile = 0;
                            first_thread_pos_in_tile < second_arr_buff_prev_used;
                                                first_thread_pos_in_tile += blockDim.x) {

                     if (first_thread_pos_in_tile + threadIdx.x <
                              second_arr_len_per_tblock - second_arr_loaded_total &&
                                  first_thread_pos_in_tile + threadIdx.x < second_arr_buff_prev_used) {

                          second_array_buff[(second_arr_buff_write_start +
                                                 threadIdx.x + first_thread_pos_in_tile) %
                                                                               data_buff_len_per_arr] =
                                second_input_array[second_arr_tblock_start + second_arr_loaded_total +
                                                                     threadIdx.x + first_thread_pos_in_tile];                     

                     }

               }

               /* 
                * No need to advance indicies right now to keep track of current write positions 
                * in shared memory since they will be updated to the old read positions before next
                * iteration; however we need to remember how much we've already read from global
                * memory
                */
               first_arr_loaded_total += min(first_arr_buff_prev_used,
                                               first_arr_len_per_tblock - first_arr_loaded_total);
               second_arr_loaded_total += min(second_arr_buff_prev_used,
                                               second_arr_len_per_tblock - second_arr_loaded_total);

               __syncthreads();
               
               long dest_arr_thread_start = min(threadIdx.x * (data_buff_len_per_arr / blockDim.x),
                                                          dest_arr_len_per_tblock - dest_arr_completed);
               long dest_arr_thread_end = min((threadIdx.x + 1) * (data_buff_len_per_arr / blockDim.x),
                                                            dest_arr_len_per_tblock - dest_arr_completed);
               long first_arr_virt_thread_start = determine_virt_corank_circular(dest_arr_thread_start,
                                                                                first_array_buff,
                                                                                min(data_buff_len_per_arr,
                                                                                     first_arr_len_per_tblock -
                                                                                              first_arr_processed),
                                                                                second_array_buff,
                                                                                min(data_buff_len_per_arr,
                                                                                     second_arr_len_per_tblock -
                                                                                              second_arr_processed),
                                                                                first_arr_buff_read_start,
                                                                                second_arr_buff_read_start,
                                                                                                data_buff_len_per_arr,
                                                                                                sort_non_descending);
               long first_arr_virt_thread_end = determine_virt_corank_circular(dest_arr_thread_end,
                                                                              first_array_buff,
                                                                              min(data_buff_len_per_arr,
                                                                                   first_arr_len_per_tblock -
                                                                                              first_arr_processed),
                                                                              second_array_buff,
                                                                              min(data_buff_len_per_arr,
                                                                                   second_arr_len_per_tblock -
                                                                                            second_arr_processed),
                                                                              first_arr_buff_read_start,
                                                                              second_arr_buff_read_start,
                                                                                                data_buff_len_per_arr,
                                                                                                sort_non_descending);
               long second_arr_virt_thread_start = dest_arr_thread_start - first_arr_virt_thread_start;
               long second_arr_virt_thread_end = dest_arr_thread_end - first_arr_virt_thread_end;
               
               merge_circular(first_array_buff, second_array_buff,
                              &dest_array[dest_arr_tblock_start + dest_arr_completed + dest_arr_thread_start],
                              first_arr_virt_thread_end - first_arr_virt_thread_start,
                              second_arr_virt_thread_end - second_arr_virt_thread_start,
                              first_arr_buff_read_start + first_arr_virt_thread_start,
                                  second_arr_buff_read_start + second_arr_virt_thread_start, data_buff_len_per_arr,
                                                                                                sort_non_descending);

               first_arr_buff_prev_used =
                     determine_virt_corank_circular(min(data_buff_len_per_arr, dest_arr_len_per_tblock - dest_arr_completed),
                                                    first_array_buff,
                                                    min(data_buff_len_per_arr, first_arr_len_per_tblock - first_arr_processed),
                                                    second_array_buff,
                                                    min(data_buff_len_per_arr, second_arr_len_per_tblock - second_arr_processed),
                                                          first_arr_buff_read_start, second_arr_buff_read_start, data_buff_len_per_arr,
                                                                                                                sort_non_descending);
               second_arr_buff_prev_used = min(data_buff_len_per_arr,
                                                  dest_arr_len_per_tblock - dest_arr_completed) - first_arr_buff_prev_used;
               first_arr_processed += first_arr_buff_prev_used;
               dest_arr_completed += min(data_buff_len_per_arr, dest_arr_len_per_tblock - dest_arr_completed);
               second_arr_processed = dest_arr_completed - first_arr_processed;

               // Old "read_start" indicies now point to beginning of emptied space in circular buffers
               first_arr_buff_write_start = first_arr_buff_read_start;
               second_arr_buff_write_start = second_arr_buff_read_start;
               // Advance "read_start" indicies by how much has been emptied from each buffer
               first_arr_buff_read_start = (first_arr_buff_read_start + first_arr_buff_prev_used) % data_buff_len_per_arr;
               second_arr_buff_read_start = (second_arr_buff_read_start + second_arr_buff_prev_used) % data_buff_len_per_arr;

               __syncthreads();

       }


}

#endif

