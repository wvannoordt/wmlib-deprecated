#include "model_algebraic.h"
#include "model_ode.h"
#include "wm_support_functions.h"
#include "model_ode_energy_equation.h"
#include "indexing.h"
#include "num_lin_alg.h"
#include "wm_common.h"
#include <iostream>
#include "debug_output.h"
#include "visualization.h"
#include "wm_io.h"
#include "KernelData.h"
#if(__cpu)
#include <cmath>
using namespace std;
#endif
namespace HYBRID
{
    __commontemplate void build_linear_system_mom_turb_ode<__hybrid, 2>(const int&, LINSYS_block_tridiag<__buffertype*, 2>*);
    __commontemplate void build_linear_system_mom_turb_ode<__hybrid, 3>(const int&, LINSYS_block_tridiag<__buffertype*, 3>*);
    __commonex(const int NUM_EQS) void build_linear_system_mom_turb_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system)
    {
        //MATRIX ORDERING:
        //lhs         rhs
        //0  1      = 0
        //2  3      = 1

        //will likely need to think about caching here.
        for (int i = 0; i < NUM_EQS; i++)
        {
            linear_system->rhs[i].base = globaldata.buffer.solution.rhs[i].base + (WM_NRAY-2)*buffer_index;
        }

        for (int i = 0; i < NUM_EQS; i++)
        {
            for (int j = 0; j < NUM_EQS; j++)
            {
                linear_system->block_matrices[j+i*NUM_EQS].diag.base = globaldata.buffer.solution.diags[i+j*NUM_EQS].base     + (WM_NRAY-2)*buffer_index;
                linear_system->block_matrices[j+i*NUM_EQS].sub.base  = globaldata.buffer.solution.subdiags[i+j*NUM_EQS].base  + (WM_NRAY-3)*buffer_index;
                linear_system->block_matrices[j+i*NUM_EQS].sup.base  = globaldata.buffer.solution.supdiags[i+j*NUM_EQS].base  + (WM_NRAY-3)*buffer_index;
                linear_system->block_matrices[j+i*NUM_EQS].dim = WM_NRAY-2;
            }
        }

        linear_system->blockdim = WM_NRAY-2;
    }

    __commontemplate void compute_solution_MODEL_ODE<__hybrid, 2>(const int&);
    __commontemplate void compute_solution_MODEL_ODE<__hybrid, 3>(const int&);
    __commonex(const int NUM_EQS) void compute_solution_MODEL_ODE(const int& buffer_index)
	{
        bool laminar_from_sensor = globaldata.settings.enable_transition_sensor && (INIDX(sensor_val)<globaldata.settings.transition_sensor_threshold);
        __buffertype epsilon = 10000;
        int k = 0;

        if (laminar_from_sensor)
        {
            int idx;
            for (idx = 0; idx < WM_NRAY; idx++)
            {
                BUFFERVARIDX(u, idx) = INIDX(u) * BUFFERVARIDX(d, idx) / (INIDX(distance));
                BUFFERVARIDX(turb, idx) = 0.0;
                epsilon = 0.0;
            }
        }
        else
        {

            //pass ray tip data
            BUFFERVARIDX(u, WM_NRAY-1) = INIDX(u);
            BUFFERVARIDX(turb, WM_NRAY-1) = INIDX(turb);
            if (globaldata.settings.include_energy_equation) BUFFERVARIDX(T, WM_NRAY-1) = INIDX(T);

            LINSYS_block_tridiag<__buffertype*, NUM_EQS> linear_system;
            build_linear_system_mom_turb_ode<__hybrid, NUM_EQS>(buffer_index, &linear_system);

            //NOTE: It may speed things up to share the jacobian among groups of threads. See https://devblogs.nvidia.com/using-shared-memory-cuda-cc/

            //On devices of compute capability 2.x and 3.x, each multiprocessor
            //has 64KB of on-chip memory that can be partitioned between L1 cache and
            //shared memory. For devices of compute capability 2.x, there are two
            //settings, 48KB shared memory / 16KB L1 cache, and 16KB shared
            //memory / 48KB L1 cache. By default the 48KB shared memory setting is used.

            //Newton iteration:
            //compute rhs, i.e. F(X)
            //compute jacobian, i.e. J(X)
            //solve J(X)dX = F(X)
            //update X -> X-dX
            //repeat

            //__buffertype epsilon = 1e-50;//10000;
            __buffertype growth = 1.1;
            __buffertype relaxation_factor = 0.2;

            while (std::abs(epsilon)> globaldata.settings.error_tolerance && k++ < globaldata.settings.max_iterations)
            {
                //Relaxation factor applied to production and destruction terms in the turbulence model (numerical jacobian)
                relaxation_factor *= growth;
                if (relaxation_factor >= 1.0) relaxation_factor = 1.0;

                // Compute right-hand-side and (analytical or numerical) jacobian
                compute_rhs_lhs_ode<__hybrid, NUM_EQS>(buffer_index, &linear_system, relaxation_factor);

                //Solve using one-way coupling
                //NOTE: after solve, the solution is stored in linearsystem.rhs[eq_index].base[index].
                solve_one_way_coupled<__hybrid, NUM_EQS>(&linear_system);

                //Error is norm of update vector
                epsilon = solution_norm<__hybrid, NUM_EQS>(linear_system.rhs, WM_NRAY-2);

                //Kill the solve if epsilon is NaN or too big.
                if (abs(epsilon) > 1e7 || epsilon != epsilon)
                {
                    CPUDEBUGOUT("Detected divergence in wall model.");
                    CPUKILL;
                }

                //solution update
                update_solution_ode<__hybrid, NUM_EQS>(buffer_index, &linear_system);
            }

//            if (std::abs(epsilon) > globaldata.settings.error_tolerance)
//            {
//                init_solution_MODEL_ODE<__hybrid, NUM_EQS>(buffer_index, WMODE_INIT_WHATEVER);
//            }
        }
        OUTIDX(iterations) = (double)k;
        OUTIDX(residual) = abs(epsilon);
		populate_output_buffers<__hybrid>(buffer_index);
	}

    __commontemplate void update_solution_ode<__hybrid, 2>(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, 2>* linear_system);
    __commontemplate void update_solution_ode<__hybrid, 3>(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, 3>* linear_system);
    __commonex(const int NUM_EQS) void update_solution_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system)
    {
        int idx;
        __buffertype under_relaxation = 0.4;
        for (idx = 0; idx < WM_NRAY-2; idx++) BUFFERVARIDX(u, idx+1) -= under_relaxation*(linear_system->rhs[0].base[idx]);
        for (idx = 0; idx < WM_NRAY-2; idx++)
        {
            int sign = (linear_system->rhs[1].base[idx])<0?-1:1;
            BUFFERVARIDX(turb, idx+1) -= ((abs(BUFFERVARIDX(turb, idx+1)) < abs(under_relaxation*(linear_system->rhs[1].base[idx])) ) ? sign*0.5*BUFFERVARIDX(turb, idx+1) : under_relaxation*(linear_system->rhs[1].base[idx]));
            if (NUM_EQS>2)
            {
                BUFFERVARIDX(T, idx+1) -= 0.7*under_relaxation*linear_system->rhs[2].base[idx];
            }

        }
        if (globaldata.settings.adiabatic_wall && globaldata.settings.include_energy_equation) BUFFERVARIDX(T, 0) -= 0.5*under_relaxation*(linear_system->rhs[2].base[0]);
    }

    //Computes dM/du(idx), dM/dnu(idx), dT/du(idx) and dT/dnu(idx)
    __commontemplate void eval_lin_sys_rhs_deriv<__hybrid>(__buffertype, const int&, const int, __buffertype, __buffertype*, __buffertype*, __buffertype*, __buffertype*);
    __common void eval_lin_sys_rhs_deriv(__buffertype relaxfactor, const int& buffer_index, const int idx, __buffertype dx, __buffertype* dmdm, __buffertype* dmdt, __buffertype* dtdm, __buffertype* dtdt)
    {
        //__buffertype dmdm_loc = 0;
        //__buffertype dmdt_loc = 0;
        //__buffertype dtdm_loc = 0;
        //__buffertype dtdt_loc = 0;
        __buffertype mom1, mom2, turb1, turb2;

        __buffertype mom_f, mom_b, turb_f, turb_b;
        if (idx > 0)
        {
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, dx, 0, 0, 0, 0, 0, &mom2, &turb2);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, -dx, 0, 0, 0, 0, 0, &mom1, &turb1);
            dmdm[0] = (mom2-mom1) / (2*dx);
            dtdm[0] = (turb2-turb1) / (2*dx);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, dx, 0, 0, &mom2, &turb2);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, -dx, 0, 0, &mom1, &turb1);
            dmdt[0] = (mom2-mom1) / (2*dx);
            dtdt[0] = (turb2-turb1) / (2*dx);
        }
        else
        {
            dmdm[0] = -1e100;
            dmdt[0] = -1e100;
            dtdm[0] = -1e100;
            dtdt[0] = -1e100;
        }

        eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0,  dx, 0, 0, 0, 0, &mom2, &turb2);
        eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, -dx, 0, 0, 0, 0, &mom1, &turb1);
        dmdm[1] = (mom2-mom1) / (2*dx);
        dtdm[1] = (turb2-turb1) / (2*dx);
        eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, 0,  dx, 0, &mom2, &turb2);
        eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, 0, -dx, 0, &mom1, &turb1);
        dmdt[1] = (mom2-mom1) / (2*dx);
        dtdt[1] = (turb2-turb1) / (2*dx);

        if (idx < WM_NRAY-2)
        {
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0,  dx, 0, 0, 0, &mom2, &turb2);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, -dx, 0, 0, 0, &mom1, &turb1);
            dmdm[2] = (mom2-mom1) / (2*dx);
            dtdm[2] = (turb2-turb1) / (2*dx);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, 0, 0,  dx, &mom2, &turb2);
            eval_lin_sys_rhs_perturb<__hybrid>(relaxfactor, buffer_index, idx, 0, 0, 0, 0, 0, -dx, &mom1, &turb1);
            dmdt[2] = (mom2-mom1) / (2*dx);
            dtdt[2] = (turb2-turb1) / (2*dx);
        }
        else
        {
            dmdm[2] = -1e100;
            dmdt[2] = -1e100;
            dtdm[2] = -1e100;
            dtdt[2] = -1e100;
        }

    }



    __commontemplate void eval_lin_sys_rhs_perturb<__hybrid>(__buffertype, const int&, const int, __buffertype,
        __buffertype, __buffertype, __buffertype, __buffertype, __buffertype, __buffertype*, __buffertype*);
    __common void eval_lin_sys_rhs_perturb(__buffertype relaxfactor, const int& buffer_index, const int idx,
         __buffertype du_0, __buffertype du_1, __buffertype du_2, __buffertype dn_0, __buffertype dn_1, __buffertype dn_2,
         __buffertype* mom_out, __buffertype* turb_out)
    {
        double du[3] = {du_0,du_1,du_2};
        double dnu[3] = {dn_0,dn_1,dn_2};
        compute_rhs_turb_mom<__hybrid>(buffer_index, mom_out, turb_out, du, dnu, idx, relaxfactor);
    }
    __commontemplate void compute_rhs_turb_mom<__hybrid>(const int&, __buffertype*, __buffertype*, __buffertype*, __buffertype*, int, __buffertype);
    __common void compute_rhs_turb_mom(const int& buffer_index, __buffertype* momrhs, __buffertype* turbrhs, __buffertype* du, __buffertype* dnu, int bufidx, __buffertype relaxfactor)
    {
        __buffertype lhs_diff_mom, lhs_diff_turb, P, D, diffwall;
        compute_diff<__hybrid>(buffer_index, &lhs_diff_mom, &lhs_diff_turb, bufidx, du, dnu);
        compute_prod_dest_walldiff<__hybrid>(buffer_index, &P, &D, &diffwall, bufidx, du, dnu);
        *momrhs = lhs_diff_mom;
        *turbrhs = (lhs_diff_turb + diffwall) + relaxfactor*(P - D);
    }

    //rhs of J*dX = F(X)
    __commontemplate void compute_rhs_lhs_ode<__hybrid, 2>(const int&, LINSYS_block_tridiag<__buffertype*, 2>*, __buffertype);
    __commontemplate void compute_rhs_lhs_ode<__hybrid, 3>(const int&, LINSYS_block_tridiag<__buffertype*, 3>*, __buffertype);
    __commonex(const int NUM_EQS) void compute_rhs_lhs_ode(const int& buffer_index, LINSYS_block_tridiag<__buffertype*, NUM_EQS>* linear_system, __buffertype relaxfactor)
    {
        int idx = 0;
        //loop over only interior points
        for (idx = 0; idx < WM_NRAY-2; idx++)
        {
            bool engy_eq = globaldata.settings.include_energy_equation;
            __buffertype du[3] = {0,0,0};
            __buffertype dnu[3] = {0,0,0};
            __buffertype dT[3] = {0,0,0};
            __buffertype momrhs, turbrhs;
            compute_rhs_turb_mom<__hybrid>(buffer_index, &momrhs, &turbrhs, du, dnu, idx+1, 1.0);

            linear_system->rhs[0].base[idx] = momrhs;
            linear_system->rhs[1].base[idx] = turbrhs;
            if (engy_eq)
            {
                __buffertype engy_rhs;
                compute_engy_rhs<__hybrid>(buffer_index, &engy_rhs, idx+1, dT);
                linear_system->rhs[2].base[idx] = engy_rhs;
            }

            __buffertype dmdm[3], dmdt[3], dtdm[3], dtdt[3];
            __buffertype dEdm[3], dEdt[3], dEdE[3], dmdE[3], dtdE[3];
            eval_lin_sys_rhs_deriv<__hybrid>(relaxfactor, buffer_index, idx+1, 1e-6, dmdm, dmdt, dtdm, dtdt);
            if (engy_eq) eval_lin_sys_rhs_deriv_engy<__hybrid>(relaxfactor, buffer_index, idx+1, 0.1, dmdE, dtdE, dEdm, dEdt, dEdE);

            if (idx > 0)
            {
                linear_system->block_matrices[0+0*NUM_EQS].sub.base[idx-1] = dmdm[0];
                linear_system->block_matrices[1+0*NUM_EQS].sub.base[idx-1] = dmdt[0];
                linear_system->block_matrices[0+1*NUM_EQS].sub.base[idx-1] = dtdm[0];
                linear_system->block_matrices[1+1*NUM_EQS].sub.base[idx-1] = dtdt[0];
                if (engy_eq)
                {
                    linear_system->block_matrices[2+0*NUM_EQS].sub.base[idx-1] = dmdE[0];
                    linear_system->block_matrices[2+1*NUM_EQS].sub.base[idx-1] = dtdE[0];
                    linear_system->block_matrices[0+2*NUM_EQS].sub.base[idx-1] = dEdm[0];
                    linear_system->block_matrices[1+2*NUM_EQS].sub.base[idx-1] = dEdt[0];
                    linear_system->block_matrices[2+2*NUM_EQS].sub.base[idx-1] = dEdE[0];
                }
            }

            linear_system->block_matrices[0+0*NUM_EQS].diag.base[idx] = dmdm[1];
            linear_system->block_matrices[1+0*NUM_EQS].diag.base[idx] = dmdt[1];
            linear_system->block_matrices[0+1*NUM_EQS].diag.base[idx] = dtdm[1];
            linear_system->block_matrices[1+1*NUM_EQS].diag.base[idx] = dtdt[1];
            if (engy_eq)
            {
                linear_system->block_matrices[2+0*NUM_EQS].diag.base[idx] = dmdE[1];
                linear_system->block_matrices[2+1*NUM_EQS].diag.base[idx] = dtdE[1];
                linear_system->block_matrices[0+2*NUM_EQS].diag.base[idx] = dEdm[1];
                linear_system->block_matrices[1+2*NUM_EQS].diag.base[idx] = dEdt[1];
                linear_system->block_matrices[2+2*NUM_EQS].diag.base[idx] = dEdE[1];
            }

            if (idx < WM_NRAY-3)
            {
                linear_system->block_matrices[0+0*NUM_EQS].sup.base[idx] = dmdm[2];
                linear_system->block_matrices[1+0*NUM_EQS].sup.base[idx] = dmdt[2];
                linear_system->block_matrices[0+1*NUM_EQS].sup.base[idx] = dtdm[2];
                linear_system->block_matrices[1+1*NUM_EQS].sup.base[idx] = dtdt[2];
                if (engy_eq)
                {
                    linear_system->block_matrices[2+0*NUM_EQS].sup.base[idx] = dmdE[2];
                    linear_system->block_matrices[2+1*NUM_EQS].sup.base[idx] = dtdE[2];
                    linear_system->block_matrices[0+2*NUM_EQS].sup.base[idx] = dEdm[2];
                    linear_system->block_matrices[1+2*NUM_EQS].sup.base[idx] = dEdt[2];
                    linear_system->block_matrices[2+2*NUM_EQS].sup.base[idx] = dEdE[2];
                }
            }
        }
    }

    __commontemplate void compute_diff<__hybrid>(const int&,__buffertype*, __buffertype*, int, __buffertype*, __buffertype*);
    __common void compute_diff(const int& buffer_index, __buffertype* lhs_diff_mom, __buffertype* lhs_diff_turb, int idx, __buffertype* du, __buffertype* dnu)
    {
        __buffertype u_loc[3], T_loc[3], rho_loc[3], turb_loc[3], nu_loc[3], y_loc[3], mu_loc[3], dy_ip1_im1_inv, Dforward, Dback;
        GETLOCALVARS(d,idx,y_loc);
        GETLOCALVARS(u,idx,u_loc);
        GETLOCALVARS(turb,idx,turb_loc);
        if (!globaldata.settings.include_energy_equation)
        {
            GETLOCALVARS_F(rho, rho_loc);
            GETLOCALVARS_F(mu_lam, mu_loc);
        }
        else
        {
            GETLOCALVARS(T,idx,T_loc);
            GETLOCALVARS(rho,idx,rho_loc);
            mu_loc[0] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[0]);
            mu_loc[1] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[1]);
            mu_loc[2] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[2]);
        }
        nu_loc[0] = mu_loc[0]/rho_loc[0];
        nu_loc[1] = mu_loc[1]/rho_loc[1];
        nu_loc[2] = mu_loc[2]/rho_loc[2];
        u_loc[0] += du[0];
        u_loc[1] += du[1];
        u_loc[2] += du[2];
        turb_loc[0] += dnu[0];
        turb_loc[1] += dnu[1];
        turb_loc[2] += dnu[2];



        __buffertype grad_u_back = ((u_loc[1]-u_loc[0])/(y_loc[1] - y_loc[0]));
        __buffertype grad_u_forward = ((u_loc[2]-u_loc[1])/(y_loc[2] - y_loc[1]));
        __buffertype grad_turb_back = ((turb_loc[1]-turb_loc[0])/(y_loc[1] - y_loc[0]));
        __buffertype grad_turb_forward = ((turb_loc[2]-turb_loc[1])/(y_loc[2] - y_loc[1]));
        __buffertype grad_u = 0.5*(grad_u_back+grad_u_forward);
        __buffertype grad_turb = 0.5*(grad_turb_back+grad_turb_forward);

        //*grad_u = (*grad_u_back) + ((*grad_u_forward - *grad_u_back)/(y_loc[2] - y_loc[0]))*(y_loc[1]-y_loc[0]);
        //*grad_turb = (*grad_turb_back) + ((*grad_turb_forward - *grad_turb_back)/(y_loc[2] - y_loc[0]))*(y_loc[1]-y_loc[0]);

        dy_ip1_im1_inv = 1.0 / (0.5*(y_loc[2]-y_loc[0]));
        Dforward = (0.5*(mu_loc[1] + mu_loc[2]) + 0.5*(get_mut_SA<__hybrid>(turb_loc[1], nu_loc[1], rho_loc[1]) + get_mut_SA<__hybrid>(turb_loc[2], nu_loc[2], rho_loc[2]))) / rho_loc[2];
        Dback =    (0.5*(mu_loc[0] + mu_loc[1]) + 0.5*(get_mut_SA<__hybrid>(turb_loc[0], nu_loc[0], rho_loc[0]) + get_mut_SA<__hybrid>(turb_loc[1], nu_loc[1], rho_loc[1]))) / rho_loc[1];
        *lhs_diff_mom = dy_ip1_im1_inv*(Dforward*grad_u_forward - Dback*grad_u_back);

        Dforward = (0.5*(nu_loc[1] + nu_loc[2]) + 0.5*(turb_loc[1] + turb_loc[2]));
        Dback = (0.5*(nu_loc[0] + nu_loc[1]) + 0.5*(turb_loc[0] + turb_loc[1]));
        *lhs_diff_turb = dy_ip1_im1_inv*(Dforward*grad_turb_forward - Dback*grad_turb_back) / CONST_SA_SIGMA;
    }

    __commontemplate void compute_prod_dest_walldiff<__hybrid>(const int&, __buffertype*, __buffertype*, __buffertype*, int, __buffertype*, __buffertype*);
    __common void compute_prod_dest_walldiff(const int& buffer_index, __buffertype* P, __buffertype* D, __buffertype* diffwall, int idx, __buffertype* du, __buffertype* dnu)
    {
        __buffertype u_loc[3], T_loc[3], rho_loc[3], turb_loc[3], nu_loc[3], y_loc[3], mu_loc[3];
        GETLOCALVARS(d,idx,y_loc);
        GETLOCALVARS(u,idx,u_loc);
        GETLOCALVARS(turb,idx,turb_loc);
        if (!globaldata.settings.include_energy_equation)
        {
            GETLOCALVARS_F(rho, rho_loc);
            GETLOCALVARS_F(mu_lam, mu_loc);
        }
        else
        {
            GETLOCALVARS(T,idx,T_loc);
            GETLOCALVARS(rho,idx,rho_loc);
            mu_loc[0] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[0]);
            mu_loc[1] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[1]);
            mu_loc[2] = compute_viscous_law_MU<__hybrid>(buffer_index, T_loc[2]);
        }
        nu_loc[0] = mu_loc[0]/rho_loc[0];
        nu_loc[1] = mu_loc[1]/rho_loc[1];
        nu_loc[2] = mu_loc[2]/rho_loc[2];
        u_loc[0] += du[0];
        u_loc[1] += du[1];
        u_loc[2] += du[2];
        turb_loc[0] += dnu[0];
        turb_loc[1] += dnu[1];
        turb_loc[2] += dnu[2];


        __buffertype dturbdy_in = 0.5 * (((turb_loc[2] - turb_loc[1])/(y_loc[2] - y_loc[1])) + ((turb_loc[1] - turb_loc[0])/(y_loc[1] - y_loc[0])));
        __buffertype dudy_in = 0.5 * (((u_loc[2] - u_loc[1])/(y_loc[2] - y_loc[1])) + ((u_loc[1] - u_loc[0])/(y_loc[1] - y_loc[0])));
        __buffertype turb_in = turb_loc[1];
        __buffertype y_in = y_loc[1];


        *diffwall = CONST_SA_CB2 * dturbdy_in * dturbdy_in / CONST_SA_SIGMA;
        __buffertype omega = abs(dudy_in);
        __buffertype chi = rho_loc[1]*turb_in/mu_loc[1];
        __buffertype chi3 = chi*chi*chi;
        __buffertype fv1 = chi3 / (chi3 + CONST_SA_CV1_CUBED);
        __buffertype fv2 = 1 - chi/(1+chi*fv1);
        __buffertype sbar = fv2*turb_in / (CONST_SA_KAPPA*CONST_SA_KAPPA*y_in*y_in);
        __buffertype shat = -1;

        shat = omega+sbar;
        shat = (shat<0.3*omega)?0.3*omega:shat;

        __buffertype ft2 = 0.0;
        __buffertype r = min(10.0, turb_in / (shat*CONST_SA_KAPPA*CONST_SA_KAPPA*y_in*y_in));
        __buffertype g = r+CONST_SA_CW2*(r*r*r*r*r*r - r);
        __buffertype g6 = g*g*g*g*g*g;
        __buffertype fw = g*pow(((1+CONST_SA_CW3_POW6)/(g6 + CONST_SA_CW3_POW6)),0.16666666666667);

        *P = CONST_SA_CB1*(1-ft2)*shat*turb_in;
        *D = (CONST_SA_CW1*fw - (CONST_SA_CB1/(CONST_SA_KAPPA*CONST_SA_KAPPA))*ft2)*(turb_in*turb_in/(y_in*y_in));
    }

    __commontemplate void init_solution_MODEL_ODE<__hybrid, 2>(const int&, const int);
    __commontemplate void init_solution_MODEL_ODE<__hybrid, 3>(const int&, const int);
	__commonex(const int NUM_EQS) void init_solution_MODEL_ODE(const int& buffer_index, const int init_guess_type)
	{
        int idx;
        //__buffertype bulge = 5.3;
        __buffertype bulge = 5.3;
        __buffertype robustfactor;
		init_solution_MODEL_ALGEBRAIC<__hybrid>(buffer_index);
        compute_solution_MODEL_ALGEBRAIC<__hybrid>(buffer_index);
        __buffertype u_n = BUFFERVARIDX(u, WM_NRAY-1);
        __buffertype u_f = INIDX(u);
        __buffertype fric_ratio = INIDX(distance)*OUTIDX(tau)/(INIDX(mu_lam) * u_f);
        for (idx = 0; idx < WM_NRAY; idx++)
        {
            BUFFERVARIDX(u, idx) *= u_f/u_n;
        }
        for (idx = 0; idx < WM_NRAY; idx++)
        {
            __buffertype dloc = BUFFERVARIDX(d, idx);
            __buffertype distloc = INIDX(distance);
            __buffertype x = dloc/distloc;
            switch (init_guess_type)
            {
                case WMODE_INIT_WHATEVER:
                {
                    robustfactor = bulge*INIDX(turb)*pow(x, 0.7)*pow(((1.0-(x*x))), 0.8);
                    break;
                }
                case WMODE_INIT_TAU_RATIO:
                {
                    robustfactor = 1.18*pow(fric_ratio, 0.42)*INIDX(turb)*sqrt(x*((1.0-(x*x))));
                    break;
                }
            }
            __buffertype turb_loc = INIDX(turb)*dloc/distloc;
            __buffertype utau = sqrt(OUTIDX(tau)/INIDX(rho));
            __buffertype yplus = INIDX(rho)*dloc*utau/(INIDX(mu_lam));
            __buffertype mut = CONST_SA_KAPPA * sqrt(OUTIDX(tau)*INIDX(rho)) * dloc * (1.0 - exp(-yplus/25))*(1.0 - exp(-yplus/25));
            __buffertype nu_loc = INIDX(mu_lam)/(INIDX(rho));
            SA_backsolve_turb_var<__hybrid>(&turb_loc, mut, nu_loc, INIDX(rho), 100, 1e-25);
            BUFFERVARIDX(turb, idx) = turb_loc + robustfactor;
            BUFFERVARIDX(rho, idx) = INIDX(rho);

        }
        if (globaldata.settings.include_energy_equation)
        {
            if (globaldata.settings.adiabatic_wall)
            {
                for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(T, idx) = INIDX(T);
            }
            else
            {
                for (idx = 0; idx < WM_NRAY; idx++)
                {
                    BUFFERVARIDX(T, idx) = globaldata.settings.T_wall + (INIDX(T) - globaldata.settings.T_wall)*(BUFFERVARIDX(d, idx))/(INIDX(distance));
                }
            }
        }
	}
}
