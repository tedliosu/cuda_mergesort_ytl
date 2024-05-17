
# Instructions (Linux Version of This Program)

Build dependencies include;

1. Latest complete CUDA toolkit (CUDA 12.4 as of time of writing)

2. GNU Compiler Collection

3. [Criterion](https://github.com/Snaipe/Criterion) for running the unit tests

Open a terminal interface and run:

- `make run` to run linear buffer version of program
- `make run_circ_buff` to run circular buffer version of program (runs slower than linear version
       due to implementation overly-aggressively attempting to conserve memory bandwidth; Author
       thought this version would perform better than linear buffer version but unfortunately
       the circular buffer version was designed for running fast on older GPU architectures.)
- `make tests` to run kernel unit tests for linear buffer version of program
- `make tests_circ_buff` to run kernel unit tests for circular buffer version of program
- `make profile` to run nsight compute profiling for first invocation of `global_mem_mergesort_step`
        kernel within the linear buffer version of program (nsight compute MUST be installed for
        this to work)
- `make profile_circ_buff` to run nsight compute profiling for first invocation of `global_mem_mergesort_step`
        kernel within the circular buffer version of program (nsight compute MUST be installed for
        this to work)

Note: only Linux distributions are supported for now on this main branch.

# TODO

1. Add more detailed comments in at least the `.cu` source code files

2. Maybe add more details in this README?

3. <s>Add support for sorting 64-bit integer types as compile-time feature</s> Author deems
      this not important; as this is only essentially a demo program.

4. <s>Add unit tests at least for the CUDA kernels - Author is finding this difficult;
      any outside help would be appreciated; more than willing to refactor code to
      make unit tests easier :)</s> Done on May 13 2024 :)

5. Prevent people from entering too large array sizes based on max total VRAM (total VRAM - 512 mib basically).

6. Port over application to Windows maybe?

