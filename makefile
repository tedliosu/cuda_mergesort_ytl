
.PHONY: clean all circ_buff tests tests_circ_buff

MAIN_EXE_NAME_LIN_BUFF=main
MAIN_EXE_NAME_CIRC_BUFF=$(MAIN_EXE_NAME_LIN_BUFF)_circ_buff
TESTS_EXE_NAME_LIN_BUFF=tests
TESTS_EXE_NAME_CIRC_BUFF=$(TESTS_EXE_NAME_LIN_BUFF)_circ_buff
PROF_KERN_NAME=global_mem_mergesort_step
PROF_RES_FILE_BASENAME=$(PROF_KERN_NAME)_first
PROF_RES_FILE_BASE_CIRC_BUFF=$(PROF_KERN_NAME)_first_circ_buff

MAIN_SRC_FILES := iterative_cuda_mergesort_ytl.cu iterative_cuda_mergesort_circ_buff_ytl.cu $(wildcard *.c)
TEST_SRC_FILES := $(wildcard *.cu) splitmix64.c xoshiro256starstar.c

all:
	nvcc $(MAIN_SRC_FILES) --compiler-options -O3 --debug \
		--linker-options -lm \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
		nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--relocatable-device-code=true \
		--generate-line-info --ptxas-options --verbose -o $(MAIN_EXE_NAME_LIN_BUFF)

circ_buff:
	nvcc $(MAIN_SRC_FILES) --compiler-options -DCIRC_BUFF -O3 --debug \
		--linker-options -lm \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
		nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--relocatable-device-code=true \
		--generate-line-info --ptxas-options --verbose -o $(MAIN_EXE_NAME_CIRC_BUFF)

tests:
	nvcc $(TEST_SRC_FILES) --compiler-options -O3 \
		--linker-options -lm -lcriterion \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--relocatable-device-code=true \
		--ptxas-options --verbose -o $(TESTS_EXE_NAME_LIN_BUFF)

tests_circ_buff:
	nvcc $(TEST_SRC_FILES) --compiler-options -DCIRC_BUFF -O3 \
		--linker-options -lm -lcriterion \
		--generate-code=arch=compute_$(shell nvidia-smi --query-gpu=compute_cap \
		--format=csv,noheader | tr -d "\.\n"),code=sm_$(shell \
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d "\.\n") \
		--relocatable-device-code=true \
		--ptxas-options --verbose -o $(TESTS_EXE_NAME_CIRC_BUFF)

run: clean all
	./$(MAIN_EXE_NAME_LIN_BUFF)

run_circ_buff: clean circ_buff
	./$(MAIN_EXE_NAME_CIRC_BUFF)

run_tests: clean tests
	./$(TESTS_EXE_NAME_LIN_BUFF) -j1

run_tests_circ_buff: clean tests_circ_buff
	./$(TESTS_EXE_NAME_CIRC_BUFF) -j1

profile: clean all
	$(info $(shell echo "---Please enter sudo password when prompted---"))
	sudo --set-home $(shell which ncu) --kernel-name $(PROF_KERN_NAME) \
		--launch-count 1 --section "regex:.*" --force-overwrite \
		--export $(PROF_RES_FILE_BASENAME) ./$(MAIN_EXE_NAME_LIN_BUFF)

profile_circ_buff: clean circ_buff
	$(info $(shell echo "---Please enter sudo password when prompted---"))
	sudo --set-home $(shell which ncu) --kernel-name $(PROF_KERN_NAME) \
		--launch-count 1 --section "regex:.*" --force-overwrite \
		--export $(PROF_RES_FILE_BASE_CIRC_BUFF) ./$(MAIN_EXE_NAME_CIRC_BUFF)


clean:
	$(info $(shell echo "---Please enter sudo password when prompted---"))
	sudo --set-home rm -rf $(MAIN_EXE_NAME_LIN_BUFF) $(TESTS_EXE_NAME_LIN_BUFF) \
			$(MAIN_EXE_NAME_CIRC_BUFF) $(TESTS_EXE_NAME_CIRC_BUFF) \
			$(PROF_RES_FILE_BASENAME).ncu-rep $(PROF_RES_FILE_BASE_CIRC_BUFF).ncu-rep

