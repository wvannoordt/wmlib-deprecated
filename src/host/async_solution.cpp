#include "async_solution.h"
#include "wall_model_worker.h"
#include "wm_common.h"
#include "debug_output.h"
#include "hybrid_computing.h"
#include <thread>
#include <atomic>
#include <iostream>
#include "stdlib.h"
#include <fstream>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#include "WmTestData.h"

namespace wall_model_module
{
    std::thread sentinel;
    std::atomic<bool> awaiting_solution;

    void launch_cpu_MODEL_ALGEBRAIC(HybridComputeInstance<double*> globaldata)
    {
        HYBRID::globaldata = globaldata;
        if (cpu_solution_init_required)
        {
            cpu_solution_init_required = false;
            for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
            {
                HYBRID::init_solution_MODEL_ALGEBRAIC<__hybrid>(idx);
            }
        }

        for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
        {
            HYBRID::compute_solution_MODEL_ALGEBRAIC<__hybrid>(idx);
        }
    }

    void launch_cpu_MODEL_ODE(HybridComputeInstance<double*> globaldata)
    {
        HYBRID::globaldata = globaldata;
        if (cpu_solution_init_required)
        {
            cpu_solution_init_required = false;
            if (user_settings.include_energy_equation)
            {
                for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
                {
                    HYBRID::init_solution_MODEL_ODE<__hybrid, 3>(idx, WMODE_INIT_TAU_RATIO);
                }
            }
            else
            {
                for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
                {
                    HYBRID::init_solution_MODEL_ODE<__hybrid, 2>(idx, WMODE_INIT_TAU_RATIO);
                }
            }
        }

        if (user_settings.include_energy_equation)
        {
            for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
            {
                HYBRID::compute_solution_MODEL_ODE<__hybrid, 3>(idx);
            }
        }
        else
        {
            for (int idx = 0; idx < MemHandler.cpu_allocation; idx++)
            {
                HYBRID::compute_solution_MODEL_ODE<__hybrid, 2>(idx);
            }
        }
    }



    void print_info_local(const char* info)
	{
		if (node_info.is_global_root_process && user_settings.verbose)
		{
			std::cout << "[I] " << info << std::endl;
		}
	}

    void init_async_sentinel(void)
    {
        awaiting_solution = false;
    }

    void wmm_begin_async_solution(void)
    {
        start_wall_model_solve();
    }

    void start_wall_model_solve(void)
    {
        //std::cout << "NO LONGER IN USE." << std::endl;
        //abort();
        awaiting_solution = true;
        if (user_settings.enable_transition_sensor && sensor_init_required)
        {
            wmm_init_sensor();
        }
        if (user_settings.gpu_async_mem_transfer)
        {
            print_info_local("Initializing wall model sentinel");
            sentinel = std::thread(T_async_wall_model_solve);
        }
        else
        {
            T_async_wall_model_solve();
            awaiting_solution = false;
        }
    }

    void T_async_wall_model_solve(void)
    {
#if (CUDA_ENABLE)
        solve_gpu_allocation();
#endif
        WmInstanceData instance;
        instance.is_on_gpu = false;
        instance.num_wall_points = MemHandler.cpu_allocation;
        instance.ode_convection_pg = false;

        HybridComputeInstance<double*> globaldata;
        globaldata.instance = instance;
        globaldata.buffer = cpu_buffers;
        globaldata.settings = user_settings;

        switch(user_settings.model_selection)
        {
            default:
            case MODEL_ALGEBRAIC:
            {
                launch_cpu_MODEL_ALGEBRAIC(globaldata);
                break;
            }
            case MODEL_ODE_PRESSURE_GRADIENT_CONVECTION:
            {
                globaldata.instance.ode_convection_pg = true;
                launch_cpu_MODEL_ODE(globaldata);
                break;
            }
            case MODEL_ODE_NO_PRESSURE_GRADIENT_NO_CONVECTION:
            {
                launch_cpu_MODEL_ODE(globaldata);
                break;
            }
        }

#if (CUDA_ENABLE)
        offload_gpu_solution();
#endif

        if (user_settings.enable_transition_sensor) wmm_compute_sensor();



        if (user_settings.debug_output_allow && (node_info.global_proc_info.process_count == 1))
        {
            output_solution_data_single_proc("outputBndySolution/wmlib_testcase.dat");
        }
        if (user_settings.verbose)
        {
            double eps_total_loc = 0;
            double total_its = 0;
            for (int i = 0; i < MemHandler.cpu_allocation; i++)
            {
                eps_total_loc += cpu_buffers.out.residual.base[i];
                total_its += cpu_buffers.out.iterations.base[i];
            }

            double eps_total_glob = 0;
            double its_tot_glob = 0;
            MPI_Allreduce(&eps_total_loc, &eps_total_glob, 1, MPI_DOUBLE, MPI_SUM, node_info.wallpoint_comm);
            MPI_Allreduce(&total_its, &its_tot_glob, 1, MPI_DOUBLE, MPI_SUM, node_info.wallpoint_comm);
            if (node_info.is_global_root_process)
            {
                std::cout << "[I] Wall model residual: " << eps_total_glob/(MemHandler.cpu_allocation * node_info.global_proc_info.process_count) << ", mean iterations: " << total_its/(MemHandler.cpu_allocation * node_info.global_proc_info.process_count) << std::endl;
            }

        }


    }

    bool output_solution_data_single_proc(const char* filename)
    {
            WmTestData dataset(cpu_buffers.in, cpu_buffers.out, MemHandler.cpu_allocation, user_settings);
            dataset.write_to_file(filename);
    }

    void wmm_await_wall_model_solution(void)
    {
        await_wall_model_solution();
        MemHandler.free_inits_buffers();
    }

    void await_wall_model_solution(void)
    {
        if (awaiting_solution && user_settings.gpu_async_mem_transfer)
        {
            print_info_local("Awaiting wall model sentinel");
            sentinel.join();
            awaiting_solution = false;
            print_info_local("Solved wall model");
        }
    }
}
