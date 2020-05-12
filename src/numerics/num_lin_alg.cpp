#include <iostream>
#include "lin_alg_types.h"
#include "num_lin_alg.h"
#include "wm_lib_typedef.h"
#include "hybrid_computing.h"
#include "debug_output.h"
#if(__cpu)
#include <cmath>
using std::sqrt;
#endif

namespace HYBRID
{
    __commontemplate void solve_one_way_coupled<__hybrid, 2>(LINSYS_block_tridiag<__buffertype*, 2>*);
    __commontemplate void solve_one_way_coupled<__hybrid, 3>(LINSYS_block_tridiag<__buffertype*, 3>*);
    __commonex(const int NUM_EQS) void solve_one_way_coupled(LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system)
    {
        /*__buffertype delta_rhs;
        int idx;
        int Ndim = linear_system->block_matrices[0].dim;
        solve_thomas<__hybrid>(&(linear_system->block_matrices[1+1*NUM_EQS]), &(linear_system->rhs[1]));

        delta_rhs  = linear_system->block_matrices[1+0*NUM_EQS].diag.base[0]*linear_system->rhs[1].base[0];
        delta_rhs += linear_system->block_matrices[1+0*NUM_EQS].sup.base[0] *linear_system->rhs[1].base[1];
        linear_system->rhs[0].base[0] -= delta_rhs;

        for (idx = 1; idx < Ndim-1; idx++)
        {
            delta_rhs  = linear_system->block_matrices[1+0*NUM_EQS].diag.base[idx] * linear_system->rhs[1].base[idx];
            delta_rhs += linear_system->block_matrices[1+0*NUM_EQS].sup.base[idx]  * linear_system->rhs[1].base[idx+1];
            delta_rhs += linear_system->block_matrices[1+0*NUM_EQS].sub.base[idx-1]* linear_system->rhs[1].base[idx-1];
            linear_system->rhs[0].base[idx] -= delta_rhs;
        }

        delta_rhs  = linear_system->block_matrices[1+0*NUM_EQS].diag.base[Ndim-1]*linear_system->rhs[1].base[Ndim-1];
        delta_rhs += linear_system->block_matrices[1+0*NUM_EQS].sub.base[Ndim-2] *linear_system->rhs[1].base[Ndim-2];
        linear_system->rhs[0].base[Ndim-1] -= delta_rhs;

        solve_thomas<__hybrid>(&(linear_system->block_matrices[0+0*NUM_EQS]), &(linear_system->rhs[0]));*/

        __buffertype delta_rhs;
        int idx, blockrow, blockcol;
        int Ndim = linear_system->block_matrices[0].dim;
        solve_thomas<__hybrid>(&(linear_system->block_matrices[(NUM_EQS-1)+(NUM_EQS-1)*NUM_EQS]), &(linear_system->rhs[NUM_EQS-1]));
        /*for (int y = 0; y < Ndim; y++)
        {
            CPUDEBUGOUT(linear_system->block_matrices[8].diag.base[y]);
        }*/
        for (blockrow = NUM_EQS-2; blockrow>=0; blockrow--)
        {
            for (blockcol = NUM_EQS-1; blockcol>blockrow; blockcol--)
            {
                delta_rhs = 0.0;
                delta_rhs  = linear_system->block_matrices[blockcol+blockrow*NUM_EQS].diag.base[0]*linear_system->rhs[1].base[0];
                delta_rhs += linear_system->block_matrices[blockcol+blockrow*NUM_EQS].sup.base[0] *linear_system->rhs[1].base[1];
                linear_system->rhs[blockrow].base[0] -= delta_rhs;

                for (idx = 1; idx < Ndim-1; idx++)
                {
                    delta_rhs  = linear_system->block_matrices[blockcol+blockrow*NUM_EQS].diag.base[idx] * linear_system->rhs[1].base[idx];
                    delta_rhs += linear_system->block_matrices[blockcol+blockrow*NUM_EQS].sup.base[idx]  * linear_system->rhs[1].base[idx+1];
                    delta_rhs += linear_system->block_matrices[blockcol+blockrow*NUM_EQS].sub.base[idx-1]* linear_system->rhs[1].base[idx-1];
                    linear_system->rhs[blockrow].base[idx] -= delta_rhs;
                }

                delta_rhs  = linear_system->block_matrices[blockcol+blockrow*NUM_EQS].diag.base[Ndim-1]*linear_system->rhs[1].base[Ndim-1];
                delta_rhs += linear_system->block_matrices[blockcol+blockrow*NUM_EQS].sub.base[Ndim-2] *linear_system->rhs[1].base[Ndim-2];
                linear_system->rhs[blockrow].base[Ndim-1] -= delta_rhs;
            }

            solve_thomas<__hybrid>(&(linear_system->block_matrices[blockrow+blockrow*NUM_EQS]), &(linear_system->rhs[blockrow]));
        }

    }

    //Thomas algorithm for tridiagonal system. SOLUTION STORED IN THE RIGHT HAND SIDE!!!
    //Also, it is important that lhs->dim is set properly
    __commontemplate void solve_thomas<__hybrid>(MAT_tridiag<__buffertype*>*, Buffer<__buffertype*>*);
    __common void solve_thomas(MAT_tridiag<__buffertype*>* lhs, Buffer<__buffertype*>* rhs)
    {
        int idx, Ndim;
        Ndim = lhs->dim;

        for (idx = 1; idx < Ndim; idx++)
        {
            lhs->sub.base[idx-1] /= lhs->diag.base[idx-1]; //wi (store on subdiagonal)
            lhs->diag.base[idx] -= lhs->sub.base[idx-1]*lhs->sup.base[idx-1]; //bi = bi - wi*c(i-1)
            rhs->base[idx] -= lhs->sub.base[idx-1]*rhs->base[idx-1];
        }
        rhs->base[Ndim-1] /= lhs->diag.base[Ndim-1];
        for (idx = Ndim-2; idx >= 0; idx--) rhs->base[idx] = (rhs->base[idx] - lhs->sup.base[idx] * (rhs->base[idx+1]))/lhs->diag.base[idx];
    }

    __commontemplate __buffertype solution_norm<__hybrid, 2>(Buffer<__buffertype*>*, const int&);
    __commontemplate __buffertype solution_norm<__hybrid, 3>(Buffer<__buffertype*>*, const int&);
    __commonex(const int NUM_EQS) __buffertype solution_norm(Buffer<__buffertype*>* rhs, const int& dim)
    {
        int idx1, idx2;
        __buffertype normout = 0;
        __buffertype currentelem;
        __buffertype current_norm = 0;
        for (idx1 = 0; idx1 < NUM_EQS; idx1++)
        {
            for (idx2 = 0; idx2 < dim; idx2++)
            {
                currentelem = rhs[idx1].base[idx2];
                normout += currentelem*currentelem;
                current_norm += normout;
            }
            current_norm = sqrt(current_norm);
            current_norm = 0;
        }
        return sqrt(normout) / NUM_EQS;
    }
}
