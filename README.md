
# Instructions (Windows Version of This Program)

Build dependencies include;

1. Latest complete CUDA toolkit (CUDA 12.6 as of time of writing)

2. CMake minimum version 3.24

3. Visual Studio Community (version 2022 as of time of writing)

Open up this repository within Visual Studio Community as a folder, making sure that Visual Studio Community is already configured to handle CMake-based projects per [this article](https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170). Then the options to build either `main.exe` or `main_circ_buff.exe` should appear in the Startup Item dropdown in the toolbar, and the you may build and run either in either Release with Debug info mode or Debug mode.  The circular buffer version of this program runs slower than linear version due to implementation overly-aggressively attempting to conserve memory bandwidth; Author thought this version would perform better than linear buffer version but unfortunately the circular buffer version was designed for running fast on older GPU architectures.

Note: this branch has NOT been tested to be built with Linux; please refer to the `main` branch of this repository for the Linux version of this program.

# TODO

1. Add more detailed comments in at least the `.cu` source code files

2. Maybe add more details in this README?

3. <s>Add support for sorting 64-bit integer types as compile-time feature</s> Author deems
      this not important; as this is only essentially a demo program.

4. <s>Add unit tests at least for the CUDA kernels - Author is finding this difficult;
      any outside help would be appreciated; more than willing to refactor code to
      make unit tests easier :)</s> Done on May 13 2024 :)

5. <s>Prevent people from entering too large array sizes based on max total VRAM (total VRAM - 512 mib basically).</s>
      Done on May 17 2024, and didn't even have to use any special formulas :)
