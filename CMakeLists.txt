cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
# Build stuff for linear buffer version of program
project(CudaMergeSortYTLAll LANGUAGES C CUDA)

FILE(GLOB AllCSources *.c)
add_executable(CudaMergeSortYTLAll
    iterative_cuda_mergesort_ytl.cu
    iterative_cuda_mergesort_circ_buff_ytl.cu
    ${AllCSources}
)

set_target_properties(CudaMergeSortYTLAll
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES native
        OUTPUT_NAME "main"
)
target_link_libraries(CudaMergeSortYTLAll PRIVATE -lm)
target_compile_options(CudaMergeSortYTLAll PRIVATE
                       $<$<COMPILE_LANGUAGE:C>:
                       -O3
                       --debug
                       >
                       $<$<COMPILE_LANGUAGE:CUDA>:
                       --relocatable-device-code=true
                       --generate-line-info
                       --ptxas-options --verbose
                       >)


# Build stuff for circular buffer version of program
project(CudaMergeSortYTLCircBuff LANGUAGES C CUDA)

add_executable(CudaMergeSortYTLCircBuff
    iterative_cuda_mergesort_ytl.cu
    iterative_cuda_mergesort_circ_buff_ytl.cu
    ${AllCSources}
)

set_target_properties(CudaMergeSortYTLCircBuff
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES native
        OUTPUT_NAME "main_circ_buff"
)
target_link_libraries(CudaMergeSortYTLCircBuff PRIVATE -lm)
target_compile_options(CudaMergeSortYTLCircBuff PRIVATE
                       $<$<COMPILE_LANGUAGE:C>:
                       -O3
                       --debug
                       >
                       $<$<COMPILE_LANGUAGE:CUDA>:
                       -DCIRC_BUFF
                       --relocatable-device-code=true
                       --generate-line-info
                       --ptxas-options --verbose
                       >)



