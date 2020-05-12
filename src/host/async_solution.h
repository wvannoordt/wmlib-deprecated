#include "wall_model_worker.h"

namespace wall_model_module
{
    extern "C"
    {
        void wmm_begin_async_solution(void);
        void wmm_await_wall_model_solution(void);
        void launch_cpu_MODEL_ALGEBRAIC(HybridComputeInstance<double*> globaldata);
    	void launch_cpu_MODEL_ODE_NO_PRESSURE_GRADIENT_NO_CONVECTION(HybridComputeInstance<double*> globaldata);
    	void launch_cpu_MODEL_ODE_PRESSURE_GRADIENT_CONVECTION(HybridComputeInstance<double*> globaldata);
    }
    void init_async_sentinel(void);
    void start_wall_model_solve(void);
    void T_async_wall_model_solve(void);
    void await_wall_model_solution(void);
    bool output_solution_data_single_proc(const char* filename);

}
