#ifndef NUM_LIN_ALG_H
#define NUM_LIN_ALG_H
#include "lin_alg_types.h"
#include "hybrid_computing.h"

namespace HYBRID
{
        __commonex(const int NUM_EQS) void solve_one_way_coupled(LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system);
        __common void solve_thomas(MAT_tridiag<__buffertype*>* lhs, Buffer<__buffertype*>* rhs);
        __commonex(const int NUM_EQ) __buffertype solution_norm(Buffer<__buffertype*>* linear_system, const int& dim);
}

#endif
