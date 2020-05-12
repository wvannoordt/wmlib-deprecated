#include "wall_model_worker.h"
#include "wm_common.h"
#include "wall_model_options.h"
#include "stdlib.h"
#include <iostream>
#include "mpi.h"
#include "setup.h"
#include <unistd.h>
#include "wm_lib_typedef.h"
#include "NodeCommHandler.h"
#include "WmTestData.h"
#include "ProvidedVariableAssociations.h"
#include "GlobalMemHandler.h"
#include "hybrid_computing.h"
#include "debug_output.h"
#include "mettu18_sensor.h"
#include <fstream>
#include "wm_io.h"

namespace wall_model_module
{
	//externs/globals
	int processor_wall_point_count;
	int total_gpu_point_allocation;
	int total_node_gpu_point_allocation;
	int total_cpu_point_allocation;
	bool cpu_deallocation_required;
	bool gpu_deallocation_required;
	int process_count;
	int process_id;
	bool is_node_root_process;
	bool is_global_root_process;
	int* proc_wall_point_counts;
	int* node_gpu_point_counts;
	int current_node_wall_point_count;
	bool cpu_solution_init_required;
	bool gpu_solution_init_required;
	bool sensor_init_required;

	UserSettings user_settings;


	RestrictedBufferGroup<double*> cpu_buffers;
	RestrictedBufferGroup<__dev_real*> gpu_buffers;

	int wm_lib_input_variable_count;

	NodeCommHandler node_info;
	GlobalMemHandler MemHandler;
	ProvidedVariableAssociations UserVariables;

	void print_info(const char* info)
	{
		if (node_info.is_global_root_process && user_settings.verbose)
		{
			std::cout << "[I] " << info << std::endl;
		}
	}

	void wmm_init_sensor(void)
	{
		sensor_init_required = false;
		switch (user_settings.transition_sensor_type)
		{
			case SENSOR_METTU18:
			{
				mettu18_initialize();
				break;
			}
		}
	}

	void wmm_compute_sensor(void)
	{
		switch (user_settings.transition_sensor_type)
		{
			case SENSOR_METTU18:
			{
				mettu18_compute_sensor_values();
				break;
			}
		}
	}

	void wmm_set_dt(double* dt)
	{
		user_settings.time_step = *dt;
	}

	void wmm_associate_field_data(double* data_pointer, int* offset)
	{
		int ofloc = *offset;
		UserVariables.AssociateVariable("p",    data_pointer + 0*ofloc);
		UserVariables.AssociateVariable("u",    data_pointer + 1*ofloc);
		UserVariables.AssociateVariable("v",    data_pointer + 2*ofloc);
		UserVariables.AssociateVariable("w",    data_pointer + 3*ofloc);
		UserVariables.AssociateVariable("T",    data_pointer + 4*ofloc);
		UserVariables.AssociateVariable("turb", data_pointer + 5*ofloc);
	}

	void wmm_associate_variable(char* varname, double* var_pointer, int* stringlength)
	{
		char tempvarname[100] = {0};
		int len = *stringlength;
		memcpy(tempvarname, varname, len*sizeof(char));
		UserVariables.AssociateVariable(tempvarname, var_pointer);
	}

	void wmm_set_verbose(bool isverbose)
	{
		user_settings.verbose = isverbose;
	}
	void wmm_set_debug_out_allowed(bool allowed)
	{
		user_settings.debug_output_allow = allowed;
	}

	void wmm_init(MPI_Comm _host_comm, bool verbose_init)
	{
		cpu_solution_init_required = true;
		gpu_solution_init_required = true;
		sensor_init_required = true;
		node_info = NodeCommHandler(_host_comm);

		//Initialize global variables. It is expected that these will be overwritten when the input file is read.
		user_settings.defaults();
		user_settings.verbose = verbose_init;

		wm_lib_input_variable_count = 1;


		print_info("Initialize GPU Wall Model Library");

#if (CUDA_ENABLE)
		DO_NODE_PROC_SEQUENTIAL(print_node_info());
#endif
	}



	void print_node_info(void)
	{
		if (node_info.is_node_root_process && user_settings.verbose)
		{
			std::cout << "[I] Node " << node_info.node_name << " (0x";
			std::cout << std::hex << node_info.processor_node_id << std::dec;
			std::cout << ") has root process " << node_info.current_node_root_process_id << std::endl;
		}
	}



	void wmm_set_wall_spacing(double wall_spacing)
	{
		user_settings.max_wall_spacing = wall_spacing;
	}

	double wmm_run_test_case(const char* test_file)
	{
		WmTestData test_data;
		test_data.read_from_file(test_file);

		user_settings = test_data.settings;
		user_settings.verbose = false;
		user_settings.debug_output_allow = false;

		UserVariables.AssociateVariableFromInputData(&test_data);

		wmm_allocate_domain(&(test_data.npts), &(user_settings.ray_point_count));

		start_wall_model_solve();
		await_wall_model_solution();

		double* x = test_data["x-coordinate"];
		double* tau = test_data["wall_shear_stress"];

		std::ofstream myfile;
		myfile.open ("wmoutput.csv");


		double error = 0;
		for (int i = 0; i < test_data.npts; i++)
		{
			myfile << x[i] << "," << cpu_buffers.out.tau.base[i] << ","  << tau[i] << "\n";
			error += ((cpu_buffers.out.tau.base[i] - tau[i]) * (cpu_buffers.out.tau.base[i] - tau[i]));
		}
		error /= test_data.npts;
		myfile.close();
		return error;
	}



	void wmm_allocate_domain(int* _processor_wall_point_count, int* _ray_point_count)
	{
		processor_wall_point_count = *_processor_wall_point_count;
		user_settings.ray_point_count = *_ray_point_count;
		MemHandler = GlobalMemHandler(&node_info, *_processor_wall_point_count, &user_settings, &UserVariables);
		proc_wall_point_counts = new int[node_info.global_proc_info.process_count];
		proc_wall_point_counts[node_info.global_proc_info.process_id] = processor_wall_point_count;
		MPI_Allgather(proc_wall_point_counts+node_info.global_proc_info.process_id, 1, MPI_INT, proc_wall_point_counts, 1, MPI_INT, node_info.global_comm);
		MPI_Barrier(node_info.global_comm);
		delete[] proc_wall_point_counts;

#if (CUDA_ENABLE)
		node_info.build_gpualloc_comm((MemHandler.gpu_allocation > 0) ? 1 : 0);
		DO_NODE_PROC_SEQUENTIAL(if (node_info.is_node_root_process && user_settings.verbose) std::cout << "[I] Node " << node_info.node_name
                                    << " (" << std::hex << node_info.processor_node_id << std::dec <<") has " <<
                                     MemHandler.node_total_gpu_allocation << " GPU-allocated wall points" << std::endl);
#endif
		node_info.build_wallpoint_comm((MemHandler.processor_wall_point_count>0) ? 1 : 0);

		MemHandler.add_RestrictedBufferGroup<double>(&cpu_buffers, user_settings.model_selection, MANAGED_CPU);
#if(CUDA_ENABLE)
		MemHandler.add_RestrictedBufferGroup<__dev_real>(&gpu_buffers, user_settings.model_selection, MANAGED_GPU);
#endif
		MemHandler.allocate_buffers<__dev_real>();
		MPI_Barrier(node_info.global_comm);
		MemHandler.output_endpoint_info();
		user_settings.num_wall_points = MemHandler.gpu_allocation;

#if (CUDA_ENABLE)
		DO_GLOBAL_PROC_SEQUENTIAL(
		if (user_settings.verbose)
		{
			std::cout << "[I] Node " << node_info.node_name << " (" << std::hex << node_info.processor_node_id << std::dec << ") process ";
			std::cout << node_info.node_proc_info.process_id;
			std::cout << " has GPU virtual global memory range " << (gpu_buffers.in.distance.base + MemHandler.gpu_io_offset);
			std::cout << " to " << (gpu_buffers.in.distance.base + MemHandler.gpu_io_offset + MemHandler.gpu_allocation - 1);
			std::cout << " (" << MemHandler.gpu_allocation << " total)" << std::endl;
		}
		)
#endif
	}


	void wmm_finalize(void)
	{
		//Deallocate, etc.
		MemHandler.free_buffers();
		node_info.Close();
		print_info("Closed wall model GPU library");

	}
}
