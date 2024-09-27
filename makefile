
.PHONY: clean all circ_buff tests tests_circ_buff

MAIN_EXE_NAME_LIN_BUFF=main
MAIN_EXE_NAME_CIRC_BUFF=$(MAIN_EXE_NAME_LIN_BUFF)_circ_buff
TESTS_EXE_NAME_LIN_BUFF=tests
TESTS_EXE_NAME_CIRC_BUFF=$(TESTS_EXE_NAME_LIN_BUFF)_circ_buff

MAIN_SRC_FILES := iterative_cuda_mergesort_ytl.hip iterative_cuda_mergesort_circ_buff_ytl.hip $(wildcard *.c)
TEST_SRC_FILES := $(wildcard *.hip) splitmix64.c xoshiro256starstar.c

all:
	hipcc $(MAIN_SRC_FILES) -O3 -gline-tables-only -lm \
		--offload-arch=$(shell rocminfo | grep gfx | head -n1 | tr -s " " | \
		cut -d" " -f3) -foffload-lto -fgpu-rdc -o $(MAIN_EXE_NAME_LIN_BUFF)

circ_buff:
	hipcc $(MAIN_SRC_FILES) -D CIRC_BUFF=1 -O3 -gline-tables-only -lm \
		--offload-arch=$(shell rocminfo | grep gfx | head -n1 | tr -s " " | \
		cut -d" " -f3) -foffload-lto -fgpu-rdc -o $(MAIN_EXE_NAME_CIRC_BUFF)

tests:
	hipcc $(TEST_SRC_FILES) -O3 -lm -lcriterion \
		--offload-arch=$(shell rocminfo | grep gfx | head -n1 | tr -s " " | \
		cut -d" " -f3) -foffload-lto -fgpu-rdc -o $(TESTS_EXE_NAME_LIN_BUFF)

tests_circ_buff:
	hipcc $(TEST_SRC_FILES) -D CIRC_BUFF=1 -O3 -lm -lcriterion \
		--offload-arch=$(shell rocminfo | grep gfx | head -n1 | tr -s " " | \
		cut -d" " -f3) -foffload-lto -fgpu-rdc -o $(TESTS_EXE_NAME_CIRC_BUFF)

run: clean all
	./$(MAIN_EXE_NAME_LIN_BUFF)

run_circ_buff: clean circ_buff
	./$(MAIN_EXE_NAME_CIRC_BUFF)

run_tests: clean tests
	./$(TESTS_EXE_NAME_LIN_BUFF) -j1

run_tests_circ_buff: clean tests_circ_buff
	./$(TESTS_EXE_NAME_CIRC_BUFF) -j1

clean:
	rm -rf $(MAIN_EXE_NAME_LIN_BUFF) $(TESTS_EXE_NAME_LIN_BUFF) \
			$(MAIN_EXE_NAME_CIRC_BUFF) $(TESTS_EXE_NAME_CIRC_BUFF)

