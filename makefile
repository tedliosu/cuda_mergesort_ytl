
.PHONY: clean all

EXE_NAME=main
PROF_KERN_NAME=global_mem_mergesort_step
PROF_RES_FILE_BASENAME=$(PROF_KERN_NAME)_first

MAIN_SRC_FILES := iterative_cuda_mergesort_ytl.cu $(wildcard *.c)

all:
	nvcc $(MAIN_SRC_FILES) --compiler-options -O3 --debug \
		--linker-options -lm \
		--generate-code=arch=compute_$(shell __nvcc_device_query),code=sm_$(shell \
		__nvcc_device_query) \
		--generate-line-info --ptxas-options --verbose -o $(EXE_NAME)

run: clean all
	./$(EXE_NAME)

profile: clean all
	$(info $(shell echo "---Please enter sudo password when prompted---"))
	sudo --set-home $(shell which ncu) --kernel-name $(PROF_KERN_NAME) \
		--launch-count 1 --section "regex:.*" --force-overwrite \
		--export $(PROF_RES_FILE_BASENAME) ./$(EXE_NAME)

clean:
	rm -rf $(EXE_NAME) $(PROF_RES_FILE_BASENAME).ncu-rep

