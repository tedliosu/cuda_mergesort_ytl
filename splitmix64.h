
// Wrapper header file for splitmix64.c

#ifndef SPLITMIX64_H
#define SPLITMIX64_H

#include <stdint.h>

void set_seed_splitmix64(uint64_t seed);

uint64_t splitmix64_get_next();

#endif
