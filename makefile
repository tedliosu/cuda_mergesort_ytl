
.PHONY: clean all tests

MAIN_EXE_NAME=main
TESTS_EXE_NAME=tests
PROF_KERN_NAME=global_mem_mergesort_step
PROF_RES_FILE_BASENAME=$(PROF_KERN_NAME)_first

MAIN_SRC_FILES := iterative_cuda_mergesort_ytl.cu $(wildcard *.c)
TEST_SRC_FILES := $(wildcard *.cu) splitmix64.c xoshiro256starstar.c

all:
	nvcc $(MAIN_SRC_FILES) --compiler-options -O3 --debug \
		--linker-options -lm \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
		nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--generate-line-info --ptxas-options --verbose -o $(MAIN_EXE_NAME)

tests:
	nvcc $(TEST_SRC_FILES) --compiler-options -O3 \
		--linker-options -lm -lcriterion \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--ptxas-options --verbose -o $(TESTS_EXE_NAME)

run: clean all
	./$(MAIN_EXE_NAME)

run_tests: clean tests
	./$(TESTS_EXE_NAME) -j1

profile: clean all
	$(info $(shell echo "---Please enter sudo password when prompted---"))
	sudo --set-home $(shell which ncu) --kernel-name $(PROF_KERN_NAME) \
		--launch-count 1 --section "regex:.*" --force-overwrite \
		--export $(PROF_RES_FILE_BASENAME) ./$(MAIN_EXE_NAME)

clean:
	rm -rf $(MAIN_EXE_NAME) $(TESTS_EXE_NAME) $(PROF_RES_FILE_BASENAME).ncu-rep

