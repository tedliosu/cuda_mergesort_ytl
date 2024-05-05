
.PHONY: clean all

EXE_NAME=main

SRC_FILES := $(wildcard *.cu) $(wildcard *.c)

all:
	nvcc $(SRC_FILES) --compiler-options -O3 --debug \
		--linker-options -lm \
		--generate-code=arch=compute_$(shell __nvcc_device_query),code=sm_$(shell \
		__nvcc_device_query) \
		--generate-line-info --ptxas-options --verbose -o $(EXE_NAME)

run: clean all
	./$(EXE_NAME)

clean:
	rm -rf $(EXE_NAME)

