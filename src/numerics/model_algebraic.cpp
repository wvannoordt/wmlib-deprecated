#include "hybrid_computing.h"
#include "wm_lib_typedef.h"
#include "setup.h"
#include <iostream>
#include "wm_support_functions.h"
#include "indexing.h"
#include "wm_common.h"
#include "debug_output.h"
#include "KernelData.h"
#if(__cpu)
#include <cmath>
using namespace std;
#endif
namespace HYBRID
{
	//=======================IMPORTANT NOTES=======================
	// It is assumed that the pointers passed to compute_ functions
	// contain initial guesses. This is extremely important!!!!!!!!
	// !!!!!!!!!!!
	//
	// When indexing, the buffer index should always be the smallest
	// -stride index. This is currently handled in "indexing.h" using
	// macros. THIS MEANS THAT GLOBAL VARIABLE NAMES CANNOT CHANGE,
	// NO MATTER THE CONTEXT. This is not ideal, and a function-based
	// approach should be implemented later.
	//
	//Dear past Will, you are a fucking idiot.
	//  - future Will

	__commontemplate void compute_solution_MODEL_ALGEBRAIC<__hybrid>(const int&);
	__commontemplate void init_solution_MODEL_ALGEBRAIC<__hybrid>(const int&);

	__common void compute_solution_MODEL_ALGEBRAIC(const int& buffer_index)
	{
		int idx = 0;
		__buffertype epsilon = 100;
		__buffertype its = 0.0;
		bool laminar_from_sensor = globaldata.settings.enable_transition_sensor && (INIDX(sensor_val)<globaldata.settings.transition_sensor_threshold);
		if (laminar_from_sensor)
        {
            for (idx = 0; idx < WM_NRAY; idx++)
            {
                BUFFERVARIDX(u, idx) = INIDX(u) * BUFFERVARIDX(d, idx) / (INIDX(distance));
                epsilon = 0.0;
            }
        }
		else
		{
			#pragma omp simd
			for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(u, idx) = INIDX(u)*BUFFERVARIDX(d, idx)/INIDX(distance);
			__buffertype u_tau = sqrt( (INIDX(mu_lam)/INIDX(rho)) * (BUFFERVARIDX(u, 2)/BUFFERVARIDX(d, 2)));
			algebraic_model_newton_uTau<__hybrid>(buffer_index, &u_tau, &epsilon, &its);
			#pragma omp simd
			for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(yplus, idx) = BUFFERVARIDX(d,idx) * u_tau / BUFFERVARIDX(nu,idx);
			for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(uplus, idx) = get_uplus_algebraic<__hybrid>(BUFFERVARIDX(yplus, idx));
			for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(u, idx) = BUFFERVARIDX(uplus,idx) * u_tau;
		}
		OUTIDX(iterations) = its;
        OUTIDX(residual) = abs(epsilon);
		populate_output_buffers<__hybrid>(buffer_index);
	}

	__common void init_solution_MODEL_ALGEBRAIC(const int& buffer_index)
	{
		//This doesn't need to be as optimized as the solver functions.
		int idx = 0;

		//Initialize solution output buffers.
		for (idx = 0; idx < WM_NRAY; idx++)
		{
			BUFFERVARIDX(d, idx) = 1.0;
		}

		__buffertype growth_rate = 2.0; //might be worth doing some analysis here!!! <- not at all
		get_growth_rate<__hybrid>(&growth_rate, 0.0, INIDX(distance), globaldata.settings.max_wall_spacing, globaldata.settings.ray_point_count, globaldata.settings.error_tolerance, globaldata.settings.max_iterations);


		__buffertype mu_t_end = get_mut_SA<__hybrid>(INIDX(turb), INIDX(mu_lam), INIDX(rho));
		BUFFERVARIDX(d, 0)      = 0.0;
		BUFFERVARIDX(d, 1)      = globaldata.settings.max_wall_spacing;
		BUFFERVARIDX(d, WM_NRAY-1) = INIDX(distance);

		for (idx = 2; idx < WM_NRAY-1; idx++) BUFFERVARIDX(d, idx) = BUFFERVARIDX(d, idx-1) + growth_rate*(BUFFERVARIDX(d, idx-1)-BUFFERVARIDX(d, idx-2));

		#pragma omp simd
		for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(u, idx) = INIDX(u)*BUFFERVARIDX(d, idx)/INIDX(distance);

		#pragma omp simd
		for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(nu, idx) = INIDX(mu_lam) / INIDX(rho);
		for (idx = 0; idx < WM_NRAY; idx++) BUFFERVARIDX(rho, idx) = INIDX(rho);
	}
}
