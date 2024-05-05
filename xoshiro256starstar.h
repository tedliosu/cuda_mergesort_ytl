
// Wrapper header file for xoshiro256starstar.h

#ifndef XOSHIRO256STARSTAR_H
#define XOSHIRO256STARSTAR_H

#include <stdint.h>

void init_xoshiro256starstar(void);

uint64_t xoshiro256starstar_get_next(void);

void xoshiro256starstar_jump(void);

void xoshiro256starstar_long_jump(void);

#endif
