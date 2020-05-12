#ifndef WALL_MODEL_WORKER_H
#define WALL_MODEL_WORKER_H

#include "wm_common.h"
#include "wm_kernel.h"
#include "wall_model_options.h"
#include "mpi.h"
#include "num_lin_alg.h"
#include "wm_lib_typedef.h"
#include "NodeCommHandler.h"
#include "GlobalMemHandler.h"
#include "async_solution.h"
#include "visualization.h"
#include "WmTestData.h"

namespace wall_model_module
{
	extern int processor_wall_point_count;
	extern int total_gpu_point_allocation;
	extern int total_cpu_point_allocation;
	extern int total_node_gpu_point_allocation;
	extern bool cpu_deallocation_required;
	extern bool gpu_deallocation_required;

	extern UserSettings user_settings;


	extern int* proc_wall_point_counts;
	extern int* node_gpu_point_counts;
	extern int current_node_wall_point_count;

	extern double* output_stress_tensors;
	extern double* output_vorticity_corrections;
	extern double* output_shear_stress;

	extern int wm_lib_input_variable_count;
	extern bool cpu_solution_init_required;
	extern bool gpu_solution_init_required;
	extern bool sensor_init_required;

	extern NodeCommHandler node_info;
	extern GlobalMemHandler MemHandler;
	extern RestrictedBufferGroup<double*> cpu_buffers;
	extern RestrictedBufferGroup<__dev_real*> gpu_buffers;
	extern ProvidedVariableAssociations UserVariables;

	extern "C"
	{
		void wmm_init(MPI_Comm host_comm, bool verbose_init);
		void wmm_init_sensor(void);
		double wmm_run_test_case(const char* test_file);
		void wmm_allocate_domain(int* _processor_wall_point_count, int* _ray_point_count);
		void wmm_associate_variable(char* varname, double* var_pointer, int* stringlength);
		void wmm_associate_field_data(double* data_pointer, int* offset);
		void wmm_set_debug_out_allowed(bool allowed);
		void wmm_finalize(void);
		void wmm_set_dt(double* dt);
		void print_info(const char* info);
		void wmm_compute_sensor(void);
		void print_warning(const char* info);
		void wmm_write_restart_file(char* name_in, int* name_len);
		void wmm_read_restart_file(char* name_in, int* name_len);
		void determine_is_node_root_process(void);
		void print_node_info(void);
		void wmm_set_wall_spacing(double wall_spacing);
		void wmm_set_verbose(bool isverbose);
	}
}


#endif
