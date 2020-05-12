#include "wm_common.h"
#include "wall_model_options.h"
#include "wall_model_worker.h"
#include "wm_lib_typedef.h"
#include "hybrid_computing.h"
#include <stdio.h>
#include <iostream>
#include "setup.h"
#include <stdlib.h>
#include "mpi.h"
#include <atomic>
#include <unistd.h>
#include <iostream>
#include "GlobalMemHandler.h"
namespace wall_model_module
{
#define CU_ERROR_CHECK_WMK(something) CudaKillIfError_WMK(something, __FILE__, __LINE__, &node_info)
	void CudaKillIfError_WMK(cudaError _cu_error, const char* file, const int line, NodeCommHandler* node_info)
	{
		if (_cu_error != cudaSuccess)
		{
			if (node_info->is_node_root_process)
			{
				std::cout << ">>>>>>>>>>> Fatal CUDA runtime error on node ";
				std::cout << node_info->node_name << " (" << std::hex << node_info->processor_node_id << std::dec << "), process " << node_info->global_proc_info.process_id << ":" << std::endl;
				std::cout << ">>>>>>>>>>> File: " << file << ", line " << line << std::endl;
				std::cout << ">>>>>>>>>>> Error: " << cudaGetErrorString(_cu_error) << std::endl;
				abort();
			}
		}
	}
	__device__ int compute_global_index(void)
	{
		//for now
		return threadIdx.x;
	}

	__global__ void K_pass_global(HybridComputeInstance<__dev_real*> K_globaldata)
	{
		HYBRID::globaldata = K_globaldata;
	}

	//I think this needs some restructuring!! One definition rule.
	__global__ void K_compute_solution_MODEL_ALGEBRAIC()
	{
		HYBRID::compute_solution_MODEL_ALGEBRAIC<__hybrid>(compute_global_index());
	}

	__global__ void K_init_solution_MODEL_ALGEBRAIC()
	{
		HYBRID::init_solution_MODEL_ALGEBRAIC<__hybrid>(compute_global_index());
	}



	template <const int NUM_EQS> __global__ void K_compute_solution_MODEL_ODE()
	{
		HYBRID::compute_solution_MODEL_ODE<__hybrid, NUM_EQS>(compute_global_index());
	}
	template __global__ void K_compute_solution_MODEL_ODE<2>();
	template __global__ void K_compute_solution_MODEL_ODE<3>();


	template <const int NUM_EQS> __global__ void K_init_solution_MODEL_ODE()
	{
		HYBRID::init_solution_MODEL_ODE<__hybrid, NUM_EQS>(compute_global_index(), WMODE_INIT_TAU_RATIO);
	}
	template __global__ void K_init_solution_MODEL_ODE<2>();
	template __global__ void K_init_solution_MODEL_ODE<3>();

	void launch_kernel_MODEL_ALGEBRAIC(HybridComputeInstance<__dev_real*> globaldata)
	{
		K_pass_global<<<1,1>>>(globaldata);
		if (MemHandler.node_total_gpu_allocation > 0)
		{
			MPI_Barrier(node_info.gpualloc_comm);
			MemHandler.data_transfer<double>(cudaMemcpyHostToDevice);
			MPI_Barrier(node_info.gpualloc_comm);
			if (gpu_solution_init_required && node_info.is_node_root_process)
			{
				K_init_solution_MODEL_ALGEBRAIC<<<1, MemHandler.node_total_gpu_allocation>>>();
				CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
				CU_ERROR_CHECK_WMK(cudaDeviceSynchronize());
			}
			gpu_solution_init_required = false;
			if (node_info.is_node_root_process)
			{
				K_compute_solution_MODEL_ALGEBRAIC<<<1, MemHandler.node_total_gpu_allocation>>>();
				CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
			}
		}
	}

	void launch_kernel_MODEL_ODE(HybridComputeInstance<__dev_real*> globaldata)
	{
		K_pass_global<<<1,1>>>(globaldata);
		if (MemHandler.node_total_gpu_allocation > 0)
		{
			MPI_Barrier(node_info.gpualloc_comm);
			MemHandler.data_transfer<double>(cudaMemcpyHostToDevice);
			MPI_Barrier(node_info.gpualloc_comm);
			if (user_settings.include_energy_equation)
			{
				if (gpu_solution_init_required && node_info.is_node_root_process)
				{
					K_init_solution_MODEL_ODE<2><<<1, MemHandler.node_total_gpu_allocation>>>();
					CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
					CU_ERROR_CHECK_WMK(cudaDeviceSynchronize());
				}
				gpu_solution_init_required = false;
				if (node_info.is_node_root_process)
				{
					K_compute_solution_MODEL_ODE<2><<<1, MemHandler.node_total_gpu_allocation>>>();
					CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
				}
			}
			else
			{
				if (gpu_solution_init_required && node_info.is_node_root_process)
				{
					K_init_solution_MODEL_ODE<3><<<1, MemHandler.node_total_gpu_allocation>>>();
					CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
					CU_ERROR_CHECK_WMK(cudaDeviceSynchronize());
				}
				gpu_solution_init_required = false;
				if (node_info.is_node_root_process)
				{
					K_compute_solution_MODEL_ODE<3><<<1, MemHandler.node_total_gpu_allocation>>>();
					CU_ERROR_CHECK_WMK(cudaPeekAtLastError());
				}
			}
		}
	}

	void solve_gpu_allocation(void)
	{
		if (MemHandler.gpu_allocation > 0)
		{
			WmInstanceData instance;
			instance.is_on_gpu = true;
			instance.num_wall_points = MemHandler.gpu_allocation;
			instance.ode_convection_pg = false;

			HybridComputeInstance<__dev_real*> globaldata;
	        globaldata.instance = instance;
	        globaldata.buffer = gpu_buffers;
	        globaldata.settings = user_settings;

			switch(user_settings.model_selection)
			{
				default:
				case MODEL_ALGEBRAIC:
				{
					launch_kernel_MODEL_ALGEBRAIC(globaldata);
					break;
				}
				case MODEL_ODE_PRESSURE_GRADIENT_CONVECTION:
				{
					instance.ode_convection_pg = true;
					launch_kernel_MODEL_ODE(globaldata);
					break;
				}
				case MODEL_ODE_NO_PRESSURE_GRADIENT_NO_CONVECTION:
				{
					launch_kernel_MODEL_ODE(globaldata);
					break;
				}
			}
		}
	}

	void offload_gpu_solution(void)
	{
		//MPI_Barrier(node_info.gpualloc_comm); // <-- this took ~7 hours to track down.
		if (MemHandler.node_total_gpu_allocation > 0)
		{
			CU_ERROR_CHECK_WMK(cudaDeviceSynchronize()); // THIS MIGHT BE AN ISSUE!!!!!!
			MPI_Barrier(node_info.gpualloc_comm);
			MemHandler.data_transfer<double>(cudaMemcpyDeviceToHost);
			MPI_Barrier(node_info.gpualloc_comm);
		}
	}
}
