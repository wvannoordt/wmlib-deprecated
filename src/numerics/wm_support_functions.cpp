#include "hybrid_computing.h"
#include "wall_model_options.h"
#include "wm_support_functions.h"
#include "wm_lib_typedef.h"
#include "indexing.h"
#include "debug_output.h"
#include <iostream>
#include "setup.h"
#include "KernelData.h"
#if(__cpu)
#include <cmath>
using std::pow;
using std::abs;
using std::log;
using std::atan2;
#endif

namespace HYBRID
{
    // WVN, 4 March 2020
    //
    // ==================NOTE==================================
    // These calls should be thought about carefully.
    // They are to be optimized to run on GPU hardware.
    // This often comes as a detriment to readability,
    // so it is VERY important that each function is documented.
    //
    // Guidelines:
    // 1. Minimize the number of calls to special functions (sin, cos, log, etc).
    // 2. Declare as few variables as you can. GPUs have limited memory.
    // 3. ABSOLUTELY no global state can be declared here.
    // 4. Anything known at compile time should be declared in that way.
    // 5. Additive/subtractive operations are cheap, multiplies are quite cheap, divisions are the most expensive.
    // 6. Only solution buffer entries and temporary variables are of the type __buffertype. Everything else is a double.
    __commontemplate __buffertype get_uplus_algebraic<__hybrid>(__buffertype);
    __commontemplate __buffertype get_uplus_prime_algebraic<__hybrid>(__buffertype, __buffertype, __buffertype);
    __commontemplate __buffertype get_mut_SA<__hybrid>(__buffertype, __buffertype, __buffertype);
    __commontemplate void linspace_init<__hybrid>(__buffertype*, int, __buffertype, __buffertype);
    __commontemplate void constant_init<__hybrid>(__buffertype*, int, __buffertype);
    __commontemplate void get_growth_rate<__hybrid>(__buffertype*, double, double, double, int, double, int);
    __commontemplate void algebraic_model_newton_uTau<__hybrid>(const int&, __buffertype*, __buffertype*, __buffertype*);
    __commontemplate void mat_multiply<__hybrid>(const __buffertype*, const __buffertype*, __buffertype*);
    __commontemplate void mat_multiply_and_transpose<__hybrid>(const __buffertype*, const __buffertype*, __buffertype*);
	__commontemplate void populate_output_buffers<__hybrid>(const int&);


    // C. Brehm et. al. Towards a Viscous Wall Model for Immersed Boundary Methods. AIAA SciTech 2018
    // Equation 16
    __common __buffertype get_uplus_algebraic(__buffertype y_plus)
    {
        return CONST_ALGEBRAIC_B
				+CONST_ALGEBRAIC_C1*log((y_plus+CONST_ALGEBRAIC_A1)*(y_plus+CONST_ALGEBRAIC_A1)+CONST_ALGEBRAIC_B1*CONST_ALGEBRAIC_B1)
		        -CONST_ALGEBRAIC_C2*log((y_plus+CONST_ALGEBRAIC_A2)*(y_plus+CONST_ALGEBRAIC_A2)+CONST_ALGEBRAIC_B2*CONST_ALGEBRAIC_B2)
	            -CONST_ALGEBRAIC_C3*atan2(CONST_ALGEBRAIC_B1, y_plus+CONST_ALGEBRAIC_A1)-CONST_ALGEBRAIC_C4*atan2(CONST_ALGEBRAIC_B2, y_plus+CONST_ALGEBRAIC_A2);
    }

    // Derivative of "get_uplus_algebraic"
    __common __buffertype get_uplus_prime_algebraic(__buffertype y_plus, __buffertype y, __buffertype nu)
    {
        __buffertype tmp1=(y_plus+CONST_ALGEBRAIC_A1);
	    __buffertype tmp2=(y_plus+CONST_ALGEBRAIC_A2);
	    return -CONST_ALGEBRAIC_C1*2*tmp1/(tmp1*tmp1+CONST_ALGEBRAIC_B1_SQUARED)*(y/nu)
	                   +CONST_ALGEBRAIC_C2*2*tmp2/(tmp2*tmp2+CONST_ALGEBRAIC_B2_SQUARED)*(y/nu)
	                   +CONST_ALGEBRAIC_C3/(1+(tmp1*CONST_ALGEBRAIC_B1_INV)*(tmp1*CONST_ALGEBRAIC_B1_INV))*(CONST_ALGEBRAIC_B1_INV)*(y/nu)
	                   +CONST_ALGEBRAIC_C4/(1+(tmp2*CONST_ALGEBRAIC_B2_INV)*(tmp2*CONST_ALGEBRAIC_B2_INV))*(CONST_ALGEBRAIC_B2_INV)*(y/nu);
    }

    // from SA model
    __common __buffertype get_mut_SA(__buffertype nutilda, __buffertype nu, __buffertype rho)
    {
        __buffertype chi = nutilda/nu;
		__buffertype fv1 = chi*chi*chi;
		fv1 = fv1/(fv1+CONST_SA_CV1_CUBED);
		return rho * fv1 * nutilda;
    }

    // Initialize with linear profile
    __common void linspace_init(__buffertype* array, int N, __buffertype x0, __buffertype x1)
    {
        __buffertype dx = (x1 - x0) / (N - 1);
        for (int i = 0; i < N; i++) *(array+i) = i*dx;
    }

    // Initialize with constant value
    __common void constant_init(__buffertype* array, int N, __buffertype constant_value)
    {
        for (int i = 0; i < N; i++) *(array+i) = constant_value;
    }

    // given an initial value and end value, compute growth rate for N points.
    __common void get_growth_rate(__buffertype* init_guess, double x0, double xn, double dx0, int N, double error_tolerance, int max_its)
    {
        __buffertype Z = (xn - x0) / dx0;
        __buffertype epsilon = 100;
        int k = 0;
        while (abs(epsilon) > error_tolerance && k++ < max_its)
        {
            epsilon = pow(*init_guess,(N - 1)) - Z*(*init_guess) + Z - 1.0;
            *init_guess = *init_guess - (epsilon) / ((N-1)*pow(*init_guess, (N - 2)) - Z);
        }
    }

    __commontemplate void SA_backsolve_turb_var<__hybrid>(__buffertype*, __buffertype, __buffertype, __buffertype, int, __buffertype);
    __common void SA_backsolve_turb_var(__buffertype* init_guess, __buffertype mu_t, __buffertype nu, __buffertype rho, int max_its, __buffertype error_tolerance)
    {
        __buffertype epsilon = 100;
        __buffertype a = mu_t / rho;
        __buffertype b = a*CONST_SA_CV1_CUBED*nu*nu*nu;
        __buffertype x = *init_guess;
        int k = 0;
        while (abs(epsilon) > error_tolerance && k++ < max_its)
        {
            epsilon = x*x*x*x - a*x*x*x - b;
            if (epsilon < error_tolerance) break;
            x = x - epsilon/(4*x*x*x - 3*a*x*x);
        }
        *init_guess = x;
    }

    // given an initial value and end value, compute growth rate for N points.
    __common void algebraic_model_newton_uTau(const int& buffer_index, __buffertype* init_guess, __buffertype* eps_out, __buffertype* its_out)
    {
        __buffertype epsilon = 100;
        __buffertype u_plus = 0.0;
        __buffertype y_plus = 0.0;
        __buffertype f;
        __buffertype fprime;
        int k = 0;
        while (abs(epsilon) > globaldata.settings.error_tolerance && k++ < globaldata.settings.max_iterations)
        {
            y_plus = (*init_guess) * BUFFERVARIDX(d, WM_NRAY-1) / BUFFERVARIDX(nu, WM_NRAY-1);
            u_plus = get_uplus_algebraic<__hybrid>(y_plus);

            f = (INIDX(u)/(*init_guess))-u_plus;
            fprime = -INIDX(u)/((*init_guess)*(*init_guess));
            epsilon = -f/fprime;
            *init_guess = (*init_guess) + 0.9*epsilon;
        }
        *eps_out = epsilon;
        *its_out = (__buffertype)k;
    }

    __common void populate_output_buffers(const int& buffer_index)
    {
        OUTIDX(tau) = INIDX(mu_lam) * (BUFFERVARIDX(u, 1) - BUFFERVARIDX(u, 0)) / globaldata.settings.max_wall_spacing;
        OUTIDX(vorticity) = abs((BUFFERVARIDX(u, 1) - BUFFERVARIDX(u, 0)) / globaldata.settings.max_wall_spacing);
        map_output_tensors<__hybrid>(buffer_index);
        if (globaldata.settings.include_energy_equation &&  globaldata.settings.adiabatic_wall)
        {
            CPUKILL("adiabatic wall not supported yet");
        }
        if (globaldata.settings.include_energy_equation && !globaldata.settings.adiabatic_wall)
        {
            __buffertype Pr = globaldata.settings.Prandtl;
            __buffertype Pr_t = globaldata.settings.Prandtl_turb;
            int samplept = 0;
            __buffertype mu =  get_MU_viscosity<__hybrid>(buffer_index, samplept);
            __buffertype heat_conductivity = globaldata.settings.Cp_fluid * (mu/Pr);
            OUTIDX(Qwall) = heat_conductivity * (BUFFERVARIDX(T, 1)-BUFFERVARIDX(T, 0))/(BUFFERVARIDX(d, 1)-BUFFERVARIDX(d, 0));
        }
    }

    __commontemplate __buffertype get_MU_viscosity<__hybrid>(const int&, int);
    __common __buffertype get_MU_viscosity(const int& buffer_index, int idx)
    {
        if (!globaldata.settings.include_energy_equation) return INIDX(mu_lam);
        return INIDX(mu_lam);
    }

    __commontemplate __buffertype compute_viscous_law_MU<__hybrid>(const int& , __buffertype);
    __common __buffertype compute_viscous_law_MU(const int& buffer_index, __buffertype T_in)
    {
        if (globaldata.settings.include_energy_equation)
        {
            return 1.45151376745308e-06 * (pow(T_in,1.50))/(T_in+110.40);
            //return (1.17 / 1.33) * 1.45151376745308e-06 * (pow(T_in,1.50))/(T_in+110.40);
        }
        return INIDX(mu_lam);
    }

    __commontemplate __buffertype compute_viscous_law_NU<__hybrid>(const int& , __buffertype);
    __common __buffertype compute_viscous_law_NU(const int& buffer_index, __buffertype T_in)
    {
        return compute_viscous_law_MU<__hybrid>(buffer_index, T_in)/INIDX(rho);
    }

    __commontemplate __buffertype get_NU_viscosity<__hybrid>(const int&, int);
    __common __buffertype get_NU_viscosity(const int& buffer_index, int idx)
    {
        return get_MU_viscosity<__hybrid>(buffer_index, idx) / INIDX(rho);
    }

    __commontemplate void map_output_tensors<__hybrid>(const int&);
    __common void map_output_tensors(const int& buffer_index)
    {
        bool compute_heat_flux = globaldata.settings.include_energy_equation && !globaldata.settings.adiabatic_wall;
        int idx1;
        //0  3  6
        //1  4  7
        //2  5  8
        //or
        //0  2
        //1  3

        __buffertype tau = OUTIDX(tau);
        __buffertype q;
        __buffertype u = INIDX(u);
        if (compute_heat_flux) q = OUTIDX(Qwall);
        //implement small matrix class for this one.
        __buffertype heat_flux_vec_working[WM_DIM];
        __buffertype u_vec[WM_DIM], sigTu[WM_DIM];
        __buffertype heat_flux_vec_output[WM_DIM];
        __buffertype stress_tensor_output[WM_DIM2];
        __buffertype metric_terms[WM_DIM2];
        __buffertype stress_tensor_xy_TRANSPOSE[WM_DIM2];


        for (idx1 = 0; idx1 < WM_DIM2; idx1++) stress_tensor_output[idx1] = INIDX_T1D(coord_sys, idx1);
        for (idx1 = 0; idx1 < WM_DIM2; idx1++) metric_terms[idx1] = INIDX_T1D(metric_data, idx1);

        //knowing that the tangent-normal-binormal stress tensor transpose is
        //0   tau 0
        //tau 0   0
        //0   0   0
        //and doing a columnwise matrix multiplication

#if(iins3d)
        stress_tensor_xy_TRANSPOSE[0] = tau*stress_tensor_output[3];
        stress_tensor_xy_TRANSPOSE[1] = tau*stress_tensor_output[4];
        stress_tensor_xy_TRANSPOSE[2] = tau*stress_tensor_output[5];
        stress_tensor_xy_TRANSPOSE[3] = tau*stress_tensor_output[0];
        stress_tensor_xy_TRANSPOSE[4] = tau*stress_tensor_output[1];
        stress_tensor_xy_TRANSPOSE[5] = tau*stress_tensor_output[2];
        stress_tensor_xy_TRANSPOSE[6] = 0.0;
        stress_tensor_xy_TRANSPOSE[7] = 0.0;
        stress_tensor_xy_TRANSPOSE[8] = 0.0;
        if (compute_heat_flux)
        {
            heat_flux_vec_working[0] = -q*stress_tensor_output[3];
            heat_flux_vec_working[1] = -q*stress_tensor_output[4];
            heat_flux_vec_working[2] = -q*stress_tensor_output[5];
            u_vec[0] = u*stress_tensor_output[0];
            u_vec[1] = u*stress_tensor_output[1];
            u_vec[2] = u*stress_tensor_output[2];
            vec_mult_by_mat<__hybrid>(stress_tensor_xy_TRANSPOSE, u_vec, sigTu);
            heat_flux_vec_working[0] -= sigTu[0];
            heat_flux_vec_working[1] -= sigTu[1];
            heat_flux_vec_working[2] -= sigTu[2];
        }
#else
        stress_tensor_xy_TRANSPOSE[0] = tau*stress_tensor_output[2];
        stress_tensor_xy_TRANSPOSE[1] = tau*stress_tensor_output[3];
        stress_tensor_xy_TRANSPOSE[2] = tau*stress_tensor_output[0];
        stress_tensor_xy_TRANSPOSE[3] = tau*stress_tensor_output[1];

        if (compute_heat_flux)
        {
            heat_flux_vec_working[0] = q*stress_tensor_output[2];
            heat_flux_vec_working[1] = q*stress_tensor_output[3];
            u_vec[0] = u*stress_tensor_output[0];
            u_vec[1] = u*stress_tensor_output[1];
            vec_mult_by_mat<__hybrid>(stress_tensor_xy_TRANSPOSE, u_vec, sigTu);
            heat_flux_vec_working[0] -= sigTu[0];
            heat_flux_vec_working[1] -= sigTu[1];
        }
#endif


        if (compute_heat_flux) vec_mult_by_mat<__hybrid>(metric_terms, heat_flux_vec_working, heat_flux_vec_output);
        mat_multiply_and_transpose<__hybrid>(metric_terms, stress_tensor_xy_TRANSPOSE, stress_tensor_output);


        for (idx1 = 0; idx1 < WM_DIM2; idx1++) OUTIDX_T1D(stress_tensors, idx1) = stress_tensor_output[idx1];

        if (compute_heat_flux)
        {
            for (idx1 = 0; idx1 < WM_DIM; idx1++)
            {
                OUTIDX_V(heat_flux_vector, idx1) = heat_flux_vec_output[idx1];
            }
        }

    }

    //b = A^t * x
    __commontemplate void vec_mult_by_mat_trans<__hybrid>(const __buffertype* PTR_SPEC, const __buffertype* PTR_SPEC, __buffertype* PTR_SPEC);
    __common void vec_mult_by_mat_trans(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC x, __buffertype* PTR_SPEC b)
    {
#if(iins3d)
        b[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2];
        b[1] = a[3]*x[0] + a[4]*x[1] + a[5]*x[2];
        b[2] = a[6]*x[0] + a[7]*x[1] + a[8]*x[2];
#else
        b[0] = a[0]*x[0] + a[1]*x[1];
        b[1] = a[2]*x[0] + a[3]*x[1];
#endif
    }

    //b = A * x
    __commontemplate void vec_mult_by_mat<__hybrid>(const __buffertype* PTR_SPEC, const __buffertype* PTR_SPEC, __buffertype* PTR_SPEC);
    __common void vec_mult_by_mat(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC x, __buffertype* PTR_SPEC b)
    {
#if(iins3d)
        b[0] = a[0]*x[0] + a[3]*x[1] + a[6]*x[2];
        b[1] = a[1]*x[0] + a[4]*x[1] + a[7]*x[2];
        b[2] = a[2]*x[0] + a[5]*x[1] + a[7]*x[2];
#else
        b[0] = a[0]*x[0] + a[2]*x[1];
        b[1] = a[1]*x[0] + a[3]*x[1];
#endif
    }

    __common void mat_multiply(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC b, __buffertype* PTR_SPEC result)
    {
        //nice
#if(iins3d)
        result[0] = a[0]*b[0] + a[3]*b[1] + a[6]*b[2];
        result[1] = a[1]*b[0] + a[4]*b[1] + a[7]*b[2];
        result[2] = a[2]*b[0] + a[4]*b[1] + a[8]*b[2];
        result[3] = a[0]*b[3] + a[3]*b[4] + a[6]*b[5];
        result[4] = a[1]*b[3] + a[4]*b[4] + a[7]*b[5];
        result[5] = a[2]*b[3] + a[4]*b[4] + a[8]*b[5];
        result[6] = a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
        result[7] = a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
        result[8] = a[2]*b[6] + a[4]*b[7] + a[8]*b[8];
#else
        result[0] = a[0]*b[0] + a[2]*b[1];
        result[1] = a[1]*b[0] + a[3]*b[1];
        result[2] = a[0]*b[2] + a[2]*b[3];
        result[3] = a[1]*b[2] + a[3]*b[3];
#endif
    }

    __common void mat_multiply_and_transpose(const __buffertype* PTR_SPEC a, const __buffertype* PTR_SPEC b, __buffertype* PTR_SPEC result)
    {
        //nice
#if(iins3d)
        result[0] = a[0]*b[0] + a[3]*b[1] + a[6]*b[2];
        result[3] = a[1]*b[0] + a[4]*b[1] + a[7]*b[2];
        result[6] = a[2]*b[0] + a[4]*b[1] + a[8]*b[2];
        result[1] = a[0]*b[3] + a[3]*b[4] + a[6]*b[5];
        result[4] = a[1]*b[3] + a[4]*b[4] + a[7]*b[5];
        result[7] = a[2]*b[3] + a[4]*b[4] + a[8]*b[5];
        result[2] = a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
        result[5] = a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
        result[8] = a[2]*b[6] + a[4]*b[7] + a[8]*b[8];
#else
        result[0] = a[0]*b[0] + a[2]*b[1];
        result[2] = a[1]*b[0] + a[3]*b[1];
        result[1] = a[0]*b[2] + a[2]*b[3];
        result[3] = a[1]*b[2] + a[3]*b[3];
#endif
    }
}
