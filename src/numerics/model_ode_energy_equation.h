#ifndef MODEL_ODE_ENGY_H
#define MODEL_ODE_ENGY_H

#include "wm_support_functions.h"
#include "num_lin_alg.h"
#include "hybrid_computing.h"
#include "KernelData.h"


namespace HYBRID
{
    __common void eval_lin_sys_rhs_deriv_engy(__buffertype relaxfactor, const int& buffer_index, const int idx, __buffertype dT,
        __buffertype* dmdE, __buffertype* dtdE, __buffertype* dEdM, __buffertype* dEdt, __buffertype* dEdE);
    __common void compute_engy_rhs(const int& buffer_index, __buffertype* engy_rhs, int idx, __buffertype* dT);
}
#endif
