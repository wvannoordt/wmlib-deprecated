#ifndef MODEL_ALG_H
#define MODEL_ALG_H
#include "hybrid_computing.h"
#include "wall_model_options.h"
#include "wm_lib_typedef.h"
#include "KernelData.h"
namespace HYBRID
{
	__common void compute_solution_MODEL_ALGEBRAIC(const int& buffer_index);
	__common void init_solution_MODEL_ALGEBRAIC(const int& buffer_index);
}
#endif
