#include "wall_model_worker.h"
#include "mettu18_sensor.h"
#include "debug_output.h"
#include "wall_model_options.h"
#include <string>

namespace wall_model_module
{
	double* mettu18_u;
	double* mettu18_v;
	double* mettu18_w;
	double* mettu18_rho;
	double* mettu18_mu_lam;
	double* mettu18_strain_rate;
	double* mettu18_strain_rate_avg;
	double* mettu18_k;
	double* mettu18_k_avg;
	double* mettu18_rho_avg;
	double* mettu18_mu_avg;
    double* mettu18_u_avg;
	double* mettu18_u_sq_avg;
    double* mettu18_v_avg;
	double* mettu18_v_sq_avg;
    double* mettu18_w_avg;
	double* mettu18_w_sq_avg;
    double* sensor_val;
    int point_num;
    double timestep;

    void mettu18_initialize(void)
    {
        point_num = processor_wall_point_count;

        if (point_num > 0)
        {
            print_info("Initializing Mettu18 sensor");
            mettu18_setvar("u",               &mettu18_u);
            mettu18_setvar("v",               &mettu18_v);
            mettu18_setvar("w",               &mettu18_w);
            mettu18_setvar("rho",             &mettu18_rho);
            mettu18_setvar("mu_lam",          &mettu18_mu_lam);
            mettu18_setvar("strain_rate",     &mettu18_strain_rate);
            mettu18_setvar("strain_rate_avg", &mettu18_strain_rate_avg);
            mettu18_setvar("k",               &mettu18_k);
            mettu18_setvar("k_avg",           &mettu18_k_avg);
            mettu18_setvar("rho_avg",         &mettu18_rho_avg);
            mettu18_setvar("mu_avg",          &mettu18_mu_avg);
            mettu18_setvar("u_avg",           &mettu18_u_avg);
            mettu18_setvar("u_sq_avg",        &mettu18_u_sq_avg);
            mettu18_setvar("v_avg",           &mettu18_v_avg);
            mettu18_setvar("v_sq_avg",        &mettu18_v_sq_avg);
            mettu18_setvar("w_avg",           &mettu18_w_avg);
            mettu18_setvar("w_sq_avg",        &mettu18_w_sq_avg);
            mettu18_setvar("sensor_val",      &sensor_val);

            zero_init(mettu18_k);
            zero_init(mettu18_k_avg);

            copy_avg(mettu18_u,           mettu18_u_avg);
            copy_avg(mettu18_v,           mettu18_v_avg);
            copy_avg(mettu18_w,           mettu18_w_avg);
            copy_avg(mettu18_rho,         mettu18_rho_avg);
            copy_avg(mettu18_strain_rate, mettu18_strain_rate_avg);
            copy_avg(mettu18_mu_lam,      mettu18_mu_avg);

            copy_avg_sq(mettu18_u, mettu18_u_sq_avg);
            copy_avg_sq(mettu18_v, mettu18_v_sq_avg);
            copy_avg_sq(mettu18_w, mettu18_w_sq_avg);
        }
    }
	double get_max_all_procs(double* buf)
	{
		double loc_max = -1e50;
		double glob_max;
		for (int i = 0; i < point_num; i++)
		{
			loc_max = (buf[i] > loc_max) ? buf[i] : loc_max;
		}
		MPI_Allreduce(&loc_max, &glob_max, 1, MPI_DOUBLE, MPI_MAX, node_info.wallpoint_comm);
		return glob_max;
	}

	double get_min_all_procs(double* buf)
	{
		double loc_min = 1e50;
		double glob_min;
		for (int i = 0; i < point_num; i++)
		{
			loc_min = (buf[i] < loc_min) ? buf[i] : loc_min;
		}
		MPI_Allreduce(&loc_min, &glob_min, 1, MPI_DOUBLE, MPI_MIN, node_info.wallpoint_comm);
		return glob_min;
	}

    void zero_init(double* a)
    {
        for (int i = 0; i < point_num; i++) a[i] = 0;
    }

    void copy_avg(double* b, double* a)
    {
        for (int i = 0; i < point_num; i++) a[i] = b[i];
    }

    void copy_avg_sq(double* b, double* a)
    {
        for (int i = 0; i < point_num; i++) a[i] = b[i]*b[i];
    }

    void mettu18_setvar(const char* varname, double** buf)
    {
        if (!UserVariables.Associated(varname, buf))
        {
            ASSERTKILL("mettu18 sensor cannot find variable \"" << varname << "\".\n This error likely occured because a wall model was chosen that does not support a transition sensor.");
        }
    }

    void mettu18_compute_average(double* phibar, double* phi)
    {
        for (int i = 0; i < point_num; i++)
        {
			double T =  1.41421356237 / (mettu18_strain_rate[i]+1e-30);
            double dt_T = timestep/T;
            phibar[i] = phibar[i]+dt_T*(phi[i]-phibar[i]);
        }
    }
    void mettu18_compute_average_sqaure(double* phibar, double* phi)
    {
        for (int i = 0; i < point_num; i++)
        {
			double T = 1.41421356237 / (mettu18_strain_rate[i]+1e-30);
			double dt_T = timestep/T;
            phibar[i] = phibar[i]+dt_T*((phi[i]*phi[i])-phibar[i]);
        }
    }
	int num_computes = 0;
	int num_saves = 0;
    void mettu18_compute_sensor_values(void)
    {
        timestep = user_settings.time_step;

        mettu18_compute_average(mettu18_u_avg, mettu18_u);
        mettu18_compute_average(mettu18_v_avg, mettu18_v);
        mettu18_compute_average(mettu18_w_avg, mettu18_w);
        mettu18_compute_average(mettu18_rho_avg, mettu18_rho);
        mettu18_compute_average(mettu18_strain_rate_avg, mettu18_strain_rate);
        mettu18_compute_average(mettu18_mu_avg, mettu18_mu_lam);
        mettu18_compute_average_sqaure(mettu18_u_sq_avg, mettu18_u);
        mettu18_compute_average_sqaure(mettu18_v_sq_avg, mettu18_v);
        mettu18_compute_average_sqaure(mettu18_w_sq_avg, mettu18_w);

        for (int i = 0; i < point_num; i++)
        {
            mettu18_k[i] = 0;
            mettu18_k[i] += mettu18_u_sq_avg[i];
            mettu18_k[i] += mettu18_v_sq_avg[i];
            mettu18_k[i] += mettu18_w_sq_avg[i];
            mettu18_k[i] -= mettu18_u_avg[i]*mettu18_u_avg[i];
            mettu18_k[i] -= mettu18_v_avg[i]*mettu18_v_avg[i];
            mettu18_k[i] -= mettu18_w_avg[i]*mettu18_w_avg[i];
        }

        mettu18_compute_average(mettu18_k_avg, mettu18_k);

        for (int i = 0; i < point_num; i++)
        {
            sensor_val[i] = 0.15 * mettu18_rho_avg[i]*mettu18_k_avg[i] / (mettu18_mu_avg[i] * mettu18_strain_rate_avg[i]);
        }
		double smin = get_min_all_procs(sensor_val);
		double smax = get_max_all_procs(sensor_val);
		if (node_info.is_global_root_process && user_settings.verbose)
		{
			std::cout << "[I] sensor min/max: " << smin << ", " <<  smax << std::endl;
		}
    }
}
