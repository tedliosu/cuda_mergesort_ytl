/* Iterative C program for merge sort */
/*
 *
 * This CUDA mergesort program uses code and algorithms from Dr. Steven S.
 * Lumetta, Dr. Wen-mei W. Hwu , Dr. David Kirk, Dr. Christian Siebert, and
 * Dr. Jesper Larsson Träff, as well as the iterative mergesort algorithm as
 * detailed on the https://www.geeksforgeeks.org/iterative-merge-sort/ page :)
 * Author: Yensong Ted Li
 *
 */

#include "main.h"
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "splitmix64.h"
#include "xoshiro256starstar.h"

// Seed for pseudorandom number generator that creates the array to be sorted
#define RAND_NUM_SEED 1234
// Think twice before changing any macro definition below this comment.
// #define NUM_NS_PER_SEC 1000000000.0
// #define MAX_USER_INPUT_LEN 21
// #define USER_INPUT_NUM_BASE 10

/* Driver program to test parallel mergesort */
int main() {
  bool sort_non_descending = true;
  long input_array_length = 0;
  const char sort_non_descending_req_char = 'T';
  char* fgets_return_stat = NULL;
  char user_input[MAX_USER_INPUT_LEN + 1] = {'\0'};
  char* remaining_non_ulong;

  while (fgets_return_stat == NULL) {
    // printf("WTF IS THIS: %ld.\n", get_max_arr_len_for_dev(sizeof(double)));
    printf(
        "Enter the size of the array to be sorted (BY ENTERING A SIZE YOU "
        "ALSO\n"
        "\tAGREE TO NOT LAUNCH NEW APPS ON THE PRIMARY GPU WHEN THIS PROGRAM\n"
        "\tIS RUNNING): ");
    fgets_return_stat = fgets(user_input, MAX_USER_INPUT_LEN + 1, stdin);
    // Remove any trailing newline and carriage returns
    user_input[strcspn(user_input, "\r\n")] = '\0';

    if (fgets_return_stat != NULL) {
      errno = 0;
      input_array_length =
          strtoul(user_input, &remaining_non_ulong, USER_INPUT_NUM_BASE);
      int strtol_error_status = errno;

      if (strtol_error_status != 0 || remaining_non_ulong[0] != '\0' ||
          input_array_length < 1 ||
          input_array_length > get_max_arr_len_for_dev(sizeof(double))) {
        fgets_return_stat = NULL;

        printf(
            "Error: invalid input - please press enter again to bring back up\n"
            "\tarray size prompt if neccessary.\n");
        // Clear remainder of input buffer
        int next_char = getchar();

        while (next_char != ((int)'\n') && next_char != EOF) {
          next_char = getchar();
        }
      }
    }
  }

  printf(
      "Enter anything starting with \"T\" (without the quotes) for sorting\n"
      "\tnon-descending, anything else (including nothing) for\n"
      "\tsorting non-ascending: ");
  fgets_return_stat = fgets(user_input, MAX_USER_INPUT_LEN + 1, stdin);
  // Remove any trailing newline and carriage returns
  user_input[strcspn(user_input, "\r\n")] = '\0';
  if (user_input[0] != sort_non_descending_req_char) {
    sort_non_descending = false;
  }

  set_seed_splitmix64(RAND_NUM_SEED);
  init_xoshiro256starstar();

  double* arr = (double*)malloc(input_array_length * sizeof(*arr));
  for (long index = 0; index < input_array_length; index++) {
    double temp = u64_to_double_conv(xoshiro256starstar_get_next());
    if (isnan(temp)) {
      temp = INFINITY;
    }
    arr[index] = temp;
  }

  printf("\nStarting sorting on GPU...\n");
  struct timespec start_func_call, end_func_call;
  timespec_get(&start_func_call, TIME_UTC);
  parallel_merge_sort(arr, input_array_length, sort_non_descending);
  timespec_get(&end_func_call, TIME_UTC);
  printf("Ended sorting on GPU!\n");

  bool sort_correctly = true;
  long uniq_nums_count = 0;

  for (long index = 1; index < input_array_length; ++index) {
    if (sort_non_descending && arr[index - 1] > arr[index]) {
      printf(
          "Sorting failed using parallel_merge_sort!\n"
          "Expected %lf at index %ld to be LESS THAN OR \n"
          "EQUAL TO %lf at index %ld\n",
          arr[index - 1], index - 1, arr[index], index);
      sort_correctly = false;
      break;

    } else if (!sort_non_descending && arr[index - 1] < arr[index]) {
      printf(
          "Sorting failed using parallel_merge_sort!\n"
          "Expected %lf at index %ld to be GREATER THAN OR \n"
          "EQUAL TO %lf at index %ld\n",
          arr[index - 1], index - 1, arr[index], index);
      sort_correctly = false;
      break;
    }
    // Get number of unique numbers for sanity check.
    uint64_t first_num_u64 = double_to_u64_conv(arr[index - 1]);
    uint64_t second_num_u64 = double_to_u64_conv(arr[index]);
    if (second_num_u64 != first_num_u64) {
      ++uniq_nums_count;
    }
  }

  if (sort_correctly) {
    printf("Sorting %s succeeded using parallel_merge_sort!\n",
           sort_non_descending ? "non-descending" : "non-ascending");
  }

  printf(
      "Time took to sort array of length %ld on GPU: %lf seconds.\n"
      "We ended up with %ld unique numbers.\n",
      input_array_length,
      difftime(end_func_call.tv_sec, start_func_call.tv_sec) +
          (((double)(end_func_call.tv_nsec - start_func_call.tv_nsec)) /
           NUM_NS_PER_SEC),
      uniq_nums_count);

  free(arr);

  return EXIT_SUCCESS;
}
