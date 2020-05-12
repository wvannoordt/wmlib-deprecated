#ifndef WM_LIB_TYPEDEF_H
#define WM_LIB_TYPEDEF_H

#include "stddef.h"
#include <fstream>
#include <iostream>
#include "wall_model_options.h"

#define MAX_BUFFERS 128
#define MAX_BUFFER_NAME_LEN 32


#ifndef RESTRICT_C_POINTERS
#define RESTRICT_C_POINTERS 1
#endif

#ifndef WM_DEBUG_OUTPUT
#define WM_DEBUG_OUTPUT 0
#endif

#ifndef WM_ALLOW_DEBUG_EXT
#define WM_ALLOW_DEBUG_EXT 0
#endif

#ifndef ANALYTICAL_JACOBIAN
#define ANALYTICAL_JACOBIAN 0
#endif

#if(RESTRICT_C_POINTERS)
#define PTR_SPEC __restrict__
#else
#define PTR_SPEC
#endif

#ifndef CUDA_ENABLE
#define CUDA_ENABLE 0
#endif

#ifndef BENCHMARKING_ENABLE
#define BENCHMARKING_ENABLE 0
#endif

#define MANAGED_CPU                0b00000000000000000000000000000001
#define MANAGED_GPU                0b00000000000000000000000000000010
#define GPU_ONETIME_ENDPOINT       0b00000000000000000000000000000100
#define GPU_ENDPOINT               0b00000000000000000000000000001000
#define CPU_ENDPOINT               0b00000000000000000000000000010000
#define SOLUTION_BUFFER            0b00000000000000000000000000100000
#define INIT_ONLY                  0b00000000000000000000000001000000
#define USER_CAN_PROVIDE           0b00000000000000000000000010000000
#define USER_MUST_PROVIDE          0b00000000000000000000000100000000
#define RESTART_BUFFER             0b00000000000000000000001000000000
#define CPU_ONLY                   0b00000000000000000000010000000000

#ifndef GPU_PRECISION
#define GPU_PRECISION 2
#endif

#if(GPU_PRECISION == 1)
#define WM_GPU_DOUBLE_PRECISION 0
typedef float __dev_real;
#else
#define WM_GPU_DOUBLE_PRECISION 1
typedef double __dev_real;
#endif

typedef unsigned char ubyte;

#ifdef __CUDACC__
#define __buffertype __dev_real
#else
#define __buffertype double
#endif

#ifndef USE_SHARED_MEMORY
#define USE_SHARED_MEMORY 1
#endif

#ifdef __CUDACC__

#define ON_GPU 1

#if(USE_SHARED_MEMORY)
#define __sh_mem __shared__
#else
#define __sh_mem
#endif

#else

#define __sh_mem
#define ON_GPU 0

#endif

typedef unsigned long __mpi_shared_device_pointer;

struct mpi_comm_info
{
	int process_id;
	int process_count;
};

#define fread_track(myptr, mysize, mycount, myfh) mysize*fread(myptr, mysize, mycount, myfh);
#define fwrite_track(myptr, mysize, mycount, myfh) mysize*fwrite(myptr, mysize, mycount, myfh);

struct UserSettings
{
	int model_selection;
	int operation_mode;
	bool debug_output_allow;
	int gpu_occupancy_policy;
	double error_tolerance;
	double gpu_allocation_ratio;
	int ray_point_count;
	bool gpu_async_mem_transfer;
	bool verbose;
	int max_iterations;
	double max_wall_spacing;
	bool include_energy_equation;
	double T_wall;
	bool adiabatic_wall;
	double Cp_fluid;
	double Prandtl_turb;
	double Prandtl;
	int num_wall_points;
	bool enable_transition_sensor;
	int transition_sensor_type;
	double time_step;
	double transition_sensor_threshold;

	void defaults(void)
	{
		operation_mode = WM_AS_ALLOCATED;
		gpu_occupancy_policy = WM_NO_GPU;
		verbose = true;
		gpu_async_mem_transfer = false;
		gpu_allocation_ratio = -1.0;
		error_tolerance = 1e-7;
		ray_point_count = 30;
		model_selection = MODEL_ALGEBRAIC;
		max_iterations = 1000;
		max_wall_spacing = 1e-7;
		debug_output_allow = false;
		Prandtl_turb = 0.9;
		Prandtl = 0.72;
		Cp_fluid = 1005.0;
		num_wall_points = -1;
		include_energy_equation = false;
		adiabatic_wall = false;
		T_wall = -100;
		enable_transition_sensor = false;
		transition_sensor_type = SENSOR_METTU18;
		time_step = -100;
		transition_sensor_threshold = 0.25;
	}

	void write_serialized(FILE* file_handle)
	{
		size_t write_size = 0;
		fwrite(&(write_size), sizeof(size_t), 1, file_handle);
		write_size += fwrite_track(&(model_selection),             sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(operation_mode),              sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(debug_output_allow),          sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(gpu_occupancy_policy),        sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(error_tolerance),             sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(gpu_allocation_ratio),        sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(ray_point_count),             sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(gpu_async_mem_transfer),      sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(verbose),                     sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(max_iterations),              sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(max_wall_spacing),            sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(include_energy_equation),     sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(T_wall),                      sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(adiabatic_wall),              sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(Cp_fluid),                    sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(Prandtl_turb),                sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(Prandtl),                     sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(num_wall_points),             sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(enable_transition_sensor),    sizeof(bool),   1, file_handle);
		write_size += fwrite_track(&(transition_sensor_type),      sizeof(int),    1, file_handle);
		write_size += fwrite_track(&(time_step),                   sizeof(double), 1, file_handle);
		write_size += fwrite_track(&(transition_sensor_threshold), sizeof(double), 1, file_handle);
		fseek(file_handle, -(write_size+sizeof(size_t)), SEEK_CUR);
		fwrite(&(write_size), sizeof(size_t), 1, file_handle);
		fseek(file_handle, write_size, SEEK_CUR);
	}

	void read_serialized(FILE* file_handle)
	{
		defaults();
		size_t file_provided_size, total_read_size;
		total_read_size = 0;
		fread(&(file_provided_size),       sizeof(size_t), 1, file_handle);
		total_read_size += fread_track(&(model_selection),             sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(operation_mode),              sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(debug_output_allow),          sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(gpu_occupancy_policy),        sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(error_tolerance),             sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(gpu_allocation_ratio),        sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(ray_point_count),             sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(gpu_async_mem_transfer),      sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(verbose),                     sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(max_iterations),              sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(max_wall_spacing),            sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(include_energy_equation),     sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(T_wall),                      sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(adiabatic_wall),              sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(Cp_fluid),                    sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(Prandtl_turb),                sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(Prandtl),                     sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(num_wall_points),             sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(enable_transition_sensor),    sizeof(bool),   1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(transition_sensor_type),      sizeof(int),    1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(time_step),                   sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
		total_read_size += fread_track(&(transition_sensor_threshold), sizeof(double), 1, file_handle); if (total_read_size >= file_provided_size) return;
	}
};

struct WmInstanceData
{
	bool is_on_gpu;
	int num_wall_points;
	bool ode_convection_pg;
};

template <typename REALPOINTER>
struct Buffer
{
	REALPOINTER PTR_SPEC base;
	int stride;
	size_t size;
};

struct InputBuffers
{
	Buffer<double*> dp_dx;
	Buffer<double*> coord_sys;
	Buffer<double*> x;
	Buffer<double*> metric_data;
	Buffer<double*> du_dt;
	Buffer<double*> du_dn;
	Buffer<double*> distance;
	Buffer<double*> p;
	Buffer<double*> u;
	Buffer<double*> v;
	Buffer<double*> w;
	Buffer<double*> T;
	Buffer<double*> turb;
	Buffer<double*> rho;
	Buffer<double*> mu_lam;
	Buffer<double*> strain_rate;
	Buffer<double*> strain_rate_avg;
	Buffer<double*> k;
	Buffer<double*> k_avg;
	Buffer<double*> rho_avg;
	Buffer<double*> mu_avg;
	Buffer<double*> u_avg;
	Buffer<double*> u_sq_avg;
	Buffer<double*> v_avg;
	Buffer<double*> v_sq_avg;
	Buffer<double*> w_avg;
	Buffer<double*> w_sq_avg;
	Buffer<double*> sensor_val;
};

struct OutputBuffers
{
	Buffer<double*> stress_tensors;
	Buffer<double*> tau;
	Buffer<double*> vorticity;
	Buffer<double*> Qwall;
	Buffer<double*> residual;
	Buffer<double*> heat_flux_vector;
	Buffer<double*> iterations;
};

#define MAX_NUM_EQS 3
template <typename REALPOINTER>
struct SolutionBuffers
{
	Buffer<REALPOINTER> u;
	Buffer<REALPOINTER> v;
	Buffer<REALPOINTER> d;
	Buffer<REALPOINTER> u0;
	Buffer<REALPOINTER> v0;
	Buffer<REALPOINTER> h;
	Buffer<REALPOINTER> l;
	Buffer<REALPOINTER> mut;
	Buffer<REALPOINTER> Omega;
	Buffer<REALPOINTER> nu;
	Buffer<REALPOINTER> rho;
	Buffer<REALPOINTER> Sa;
	Buffer<REALPOINTER> yplus;
	Buffer<REALPOINTER> uplus;
	Buffer<REALPOINTER> du_dy;
	Buffer<REALPOINTER> turb;
	Buffer<REALPOINTER> T;
	Buffer<REALPOINTER> diags[MAX_NUM_EQS*MAX_NUM_EQS];
	Buffer<REALPOINTER> subdiags[MAX_NUM_EQS*MAX_NUM_EQS];
	Buffer<REALPOINTER> supdiags[MAX_NUM_EQS*MAX_NUM_EQS];
	Buffer<REALPOINTER> rhs[MAX_NUM_EQS];
};

template <typename REALPOINTER>
struct RestrictedBufferGroup
{
	InputBuffers in;
	SolutionBuffers<REALPOINTER> solution;
	OutputBuffers out;

};

template <typename REALPOINTER>
struct HybridComputeInstance
{
	RestrictedBufferGroup<REALPOINTER> buffer;
	WmInstanceData instance;
	UserSettings settings;
};
#endif
