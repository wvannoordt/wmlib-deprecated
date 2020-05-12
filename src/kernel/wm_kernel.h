#ifndef WM_KERNEL_H
#define WM_KERNEL_H
#if(CUDA_ENABLE)
#include "cuda_headers.h"
#include "wall_model_options.h"
#include "wm_common.h"
#include "wm_lib_typedef.h"
namespace wall_model_module
{
	void free_kernel_pointers(void);
	void compute_offset(void);
	void solve_gpu_allocation(void);
	void offload_gpu_solution(void);
	void launch_kernel_MODEL_ALGEBRAIC();
	void launch_kernel_MODEL_ODE();
}
#endif
#endif
