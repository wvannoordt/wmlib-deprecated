#include "model_ode_energy_equation.h"
#include "indexing.h"
#include "debug_output.h"
#include "wm_support_functions.h"
#include <iostream>
#include "model_ode.h"
#include "hybrid_computing.h"
#include "KernelData.h"

namespace HYBRID
{
    __commontemplate void compute_engy_rhs<__hybrid>(const int&, __buffertype*, int, __buffertype*);
    __common void compute_engy_rhs(const int& buffer_index, __buffertype* engy_rhs, int idx, __buffertype* dT)
    {
        __buffertype u_loc[3], T_loc[3], rho_loc[3], turb_loc[3], nu_loc[3], y_loc[3], mu_loc[3];
        GETLOCALVARS(d,idx,y_loc);
        GETLOCALVARS(u,idx,u_loc);
        GETLOCALVARS(turb,idx,turb_loc);
        GETLOCALVARS(T,idx,T_loc);
        GETLOCALVARS(rho,idx,rho_loc);
        mu_loc[0] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[0]);
        mu_loc[1] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[1]);
        mu_loc[2] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[2]);
        nu_loc[0] = mu_loc[0]/rho_loc[0];
        nu_loc[1] = mu_loc[1]/rho_loc[1];
        nu_loc[2] = mu_loc[2]/rho_loc[2];
        T_loc[0] += dT[0];
        T_loc[1] += dT[1];
        T_loc[2] += dT[2];

        __buffertype mu_1, mu_2, mut_1, mut_2;
        __buffertype Pr = globaldata.settings.Prandtl;
        __buffertype Pr_t = globaldata.settings.Prandtl_turb;
        __buffertype prefactor = 2.0 / (y_loc[2] - y_loc[0]);

        //left face
        mu_1 = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[0]);
        mu_2 = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[1]);
        mut_1 = get_mut_SA<__hybrid>(turb_loc[0], mu_1/rho_loc[0], rho_loc[0]);
        mut_2 = get_mut_SA<__hybrid>(turb_loc[1], mu_2/rho_loc[1], rho_loc[1]);



        __buffertype diff_back_mom      = (0.5*(mu_1 + mu_2 + mut_1 + mut_2)) * (0.5*(u_loc[0]+u_loc[1])) * ((u_loc[1] - u_loc[0])/(y_loc[1] - y_loc[0]));
        __buffertype diff_back_therm    = 0.5*globaldata.settings.Cp_fluid*(mu_1/Pr + mu_2/Pr + mut_1/Pr_t + mut_2/Pr_t) * ((T_loc[1] - T_loc[0])/(y_loc[1] - y_loc[0]));


        //right face
        mu_1 = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[1]);
        mu_2 = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[2]);
        mut_1 = get_mut_SA<__hybrid>(turb_loc[1], mu_1/rho_loc[1], rho_loc[1]);
        mut_2 = get_mut_SA<__hybrid>(turb_loc[2], mu_2/rho_loc[2], rho_loc[2]);

        __buffertype diff_forward_mom   = (0.5*(mu_1 + mu_2 + mut_1 + mut_2)) * (0.5*(u_loc[1]+u_loc[2])) * ((u_loc[2] - u_loc[1])/(y_loc[2] - y_loc[1]));
        __buffertype diff_forward_therm = 0.5*globaldata.settings.Cp_fluid*(mu_1/Pr + mu_2/Pr + mut_1/Pr_t + mut_2/Pr_t) * ((T_loc[2] - T_loc[1])/(y_loc[2] - y_loc[1]));

        *engy_rhs = prefactor * (diff_forward_therm-diff_back_therm) + prefactor * (diff_forward_mom-diff_back_mom);
    }

    __commontemplate void eval_lin_sys_rhs_deriv_engy<__hybrid>(__buffertype, const int&, const int,
         __buffertype, __buffertype*, __buffertype*, __buffertype*, __buffertype*, __buffertype*);

    __common void eval_lin_sys_rhs_deriv_engy(__buffertype relaxfactor, const int& buffer_index, const int idx, __buffertype dT,
        __buffertype* dmdE, __buffertype* dtdE, __buffertype* dEdM, __buffertype* dEdt, __buffertype* dEdE)
    {

        __buffertype engy1, engy2, dT_vec[3];

        //compute energy derivatives of rhs.
        //temporary

        dT_vec[0] = 0;
        dT_vec[1] = 0;
        dT_vec[2] = 0;

        dT_vec[0] = dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy2, idx, dT_vec);
        dT_vec[0] = -dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy1, idx, dT_vec);
        dT_vec[0] = 0;

        dmdE[0] = 0.0;
        dtdE[0] = 0.0;
        dEdE[0] = (engy2-engy1)/(2*dT);

        dT_vec[1] = dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy2, idx, dT_vec);
        dT_vec[1] = -dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy1, idx, dT_vec);
        dT_vec[1] = 0;

        dmdE[1] = 0.0;
        dtdE[1] = 0.0;
        dEdE[1] = (engy2-engy1)/(2*dT);

        dT_vec[2] = dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy2, idx, dT_vec);
        dT_vec[2] = -dT;
        compute_engy_rhs<__hybrid>(buffer_index, &engy1, idx, dT_vec);
        dT_vec[2] = 0;

        dmdE[2] = 0.0;
        dtdE[2] = 0.0;
        dEdE[2] = (engy2-engy1)/(2*dT);




        dEdM[0] = 0;
        dEdM[1] = 0;
        dEdM[2] = 0;

        dEdt[0] = 0;
        dEdt[1] = 0;
        dEdt[2] = 0;

    }
}
