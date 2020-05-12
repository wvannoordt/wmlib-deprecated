#ifndef MODEL_ODE_H
#define MODEL_ODE_H

#include "wm_support_functions.h"
#include "model_algebraic.h"
#include "lin_alg_types.h"
#include "KernelData.h"

#define WMODE_INIT_TAU_RATIO 1223
#define WMODE_INIT_WHATEVER 23423

namespace HYBRID
{
    __commonex(const int NUM_EQS) void compute_solution_MODEL_ODE(const int& buffer_index);
	__commonex(const int NUM_EQS) void init_solution_MODEL_ODE(const int& buffer_index, const int init_guess_type);
    __commonex(const int NUM_EQS) void build_linear_system_mom_turb_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system);
    __commonex(const int NUM_EQS) void compute_rhs_lhs_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system, __buffertype relaxfactor);
    __commonex(const int NUM_EQS) void update_solution_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system);
    __common void compute_rhs_turb_mom(const int& buffer_index, __buffertype* momrhs, __buffertype* turbrhs, __buffertype* du, __buffertype* dnu, int bufidx, __buffertype relaxfactor);
    __common void compute_prod_dest_walldiff(const int& buffer_index, __buffertype* P, __buffertype* D, __buffertype* diffwall, int idx, __buffertype* du, __buffertype* dnu);
    __common void compute_diff(const int& buffer_index, __buffertype* lhs_diff_mom, __buffertype* lhs_diff_turb, int idx, __buffertype* du, __buffertype* dnu);
    __common void eval_lin_sys_rhs_perturb(__buffertype relaxfactor, const int& buffer_index, const int idx,
         __buffertype du_0, __buffertype du_1, __buffertype du_2, __buffertype dn_0, __buffertype dn_1, __buffertype dn_2,
         __buffertype* mom_out, __buffertype* turb_out);
    __common void eval_lin_sys_rhs_deriv(__buffertype relaxfactor, const int& buffer_index, const int idx,
         __buffertype step_size, __buffertype* dmdm, __buffertype* dmdt, __buffertype* dtdm, __buffertype* dtdt);
}

#endif
