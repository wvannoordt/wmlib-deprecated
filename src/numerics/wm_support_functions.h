#include "hybrid_computing.h"
#include "wall_model_options.h"
#include "wm_lib_typedef.h"
#include "KernelData.h"


namespace HYBRID
{
    __common __buffertype get_uplus_algebraic(__buffertype y_plus);
    __common __buffertype get_uplus_prime_algebraic(__buffertype y_plus, __buffertype y, __buffertype nu);
    __common __buffertype get_mut_SA(__buffertype nutilda, __buffertype nu, __buffertype rho);
    __common void linspace_init(__buffertype* array, int N, __buffertype x0, __buffertype x1);
    __common void constant_init(__buffertype* array, int N, __buffertype constant_value);
    __common void get_growth_rate(__buffertype* init_guess, double x0, double xn, double dx0, int N, double error_tolerance, int max_its);
    __common void algebraic_model_newton_uTau(const int& buffer_index, __buffertype* init_guess, __buffertype* eps_out, __buffertype* its_out);
	__common void map_output_tensors(const int& buffer_index);
	__common void mat_multiply(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC b, __buffertype* PTR_SPEC result);
	__common void mat_multiply_and_transpose(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC b, __buffertype* PTR_SPEC result);
	__common void populate_output_buffers(const int& buffer_index);
    __common __buffertype get_MU_viscosity(const int& buffer_index, int idx);
    __common __buffertype get_NU_viscosity(const int& buffer_index, int idx);
    __common __buffertype compute_viscous_law_MU(const int& buffer_index, __buffertype T_in);
    __common __buffertype compute_viscous_law_NU(const int& buffer_index, __buffertype T_in);
    __common void vec_mult_by_mat_trans(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC x, __buffertype* PTR_SPEC b);
    __common void vec_mult_by_mat(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC x, __buffertype* PTR_SPEC b);
    __common void SA_backsolve_turb_var(__buffertype* init_guess, __buffertype mu_t, __buffertype nu, __buffertype rho, int max_its, __buffertype error_tolerance);
}
