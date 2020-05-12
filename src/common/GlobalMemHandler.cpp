#include <stdlib.h>
#include "GlobalMemHandler.h"
#include "wall_model_options.h"
#include "wm_lib_typedef.h"
#include "NodeCommHandler.h"
#include "debug_output.h"
#include "ProvidedVariableAssociations.h"
#include "setup.h"
#if(CUDA_ENABLE)
#include "wm_kernel.h"
#include "driver_types.h"
#endif
namespace wall_model_module
{
#if(CUDA_ENABLE)
#define CU_ERROR_CHECK_GMH(myer) CudaKillIfError(myer, __FILE__, __LINE__, comm)
    void GlobalMemHandler::CudaKillIfError(cudaError _cu_error, const char* file, const int line, NodeCommHandler* node_info)
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
#endif

    template <typename my_type>
    void GlobalMemHandler::print_warning(my_type info)
    {
        if (comm->is_global_root_process && user_settings->verbose)
        {
            std::cout << "[W] " << info << std::endl;
        }
    }

    template <typename my_type>
    void GlobalMemHandler::print_info(my_type info)
    {
        if (comm->is_global_root_process && user_settings->verbose)
        {
            std::cout << "[I] " << info << std::endl;
        }
    }

    bool GlobalMemHandler::getflag(const int a, const int b)
    {
        return(a & b)>0;
    }

    GlobalMemHandler::GlobalMemHandler(){}
    GlobalMemHandler::GlobalMemHandler(NodeCommHandler* _comm, int _processor_wall_point_count, UserSettings* _user_settings, ProvidedVariableAssociations* _user_variables)
    {
        comm = _comm;
        cpu_managed_count = 0;
        gpu_managed_count = 0;

        endpoint_count = 0;
        source_count = 0;

        source_endpoint_table_index = 0;
        inits_have_been_freed = false;

        gpu_io_offset = 0;
        user_settings = _user_settings;
        processor_wall_point_count = _processor_wall_point_count;
        user_variables = _user_variables;

        restart_buffer_index = 0;
        num_cpu_restart = 0;

        get_allocation();

        cpu_count_before = 0;
        gpu_count_before = 0;
        total_count_before = 0;

        int pid = comm->global_proc_info.process_id;
        int pnum = comm->global_proc_info.process_count;

        int* temp = new int[pnum];

        temp[pid] = cpu_allocation;
        MPI_Allgather(temp + pid, 1, MPI_INT, temp, 1, MPI_INT, comm->global_comm);
        for (int i = 0; i < pid; i++) cpu_count_before+=temp[i];

        temp[pid] = gpu_allocation;
        MPI_Allgather(temp + pid, 1, MPI_INT, temp, 1, MPI_INT, comm->global_comm);
        for (int i = 0; i < pid; i++) gpu_count_before+=temp[i];

        temp[pid] = processor_wall_point_count;
        MPI_Allgather(temp + pid, 1, MPI_INT, temp, 1, MPI_INT, comm->global_comm);
        for (int i = 0; i < pid; i++) total_count_before+=temp[i];

        delete[] temp;

        if (CUDA_ENABLE) get_gpu_mem_offset();
        else gpu_io_offset = 0;
    }

    void GlobalMemHandler::ResetRestartBufferCounter(void)
    {
        restart_buffer_index = 0;
    }

    bool GlobalMemHandler::GetNextRestartBuffer(double** buffer_output)
    {
        if (restart_buffer_index>=num_cpu_restart) return false;
        *buffer_output = (double*)(*(cpu_restart[restart_buffer_index]));
        restart_buffer_index++;
        return true;
    }

    void GlobalMemHandler::free_inits_buffers(void)
    {
        if (inits_have_been_freed) return;
        else
        {
            if (cpu_allocation > 0)
            {
                for (int i = 0; i < cpu_managed_count; i++)
                {
                    if (cpu_required_to_allocate[i] && !cpu_already_freed[i] && getflag(cpu_manage_modes[i], INIT_ONLY))
                    {
                        free(*(cpu_managed[i]));
                        cpu_already_freed[i] = true;
                        if (user_settings->verbose) std::cout << "[I] Freeing init-only CPU buffer \"" << cpu_var_names[i] << "\"\n";
                    }
                }
            }
#if(CUDA_ENABLE)
            if (node_total_gpu_allocation > 0)
            {
                if (comm->is_node_root_process)
                {
                    for (int i = 0; i < gpu_managed_count; i++)
                    {
                        if (!gpu_already_freed[i] && getflag(gpu_manage_modes[i], INIT_ONLY))
                        {
                            CU_ERROR_CHECK_GMH(cudaFree(*(gpu_managed[i])));
                            gpu_already_freed[i] = true;
                            if (user_settings->verbose) std::cout << "[I] Freeing init-only GPU buffer \"" << gpu_var_names[i] << "\"\n";
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < gpu_managed_count; i++) CU_ERROR_CHECK_GMH(cudaIpcCloseMemHandle(*(gpu_managed[i])));
                }
            }
#endif
            inits_have_been_freed = true;
        }
    }

    void GlobalMemHandler::get_allocation(void)
    {
        cpu_allocation = processor_wall_point_count;
        gpu_allocation = 0;
        node_total_gpu_allocation = 0;
#if(CUDA_ENABLE)
        switch (user_settings->gpu_occupancy_policy)
        {
            case WM_CUDA_OCCUPANCY_NATIVE:
            {
                // this is a very tricky one!
                break;
            }
            case WM_USER_SPECIFIED_RATIO:
            {
                if (0.0 <= user_settings->gpu_allocation_ratio && user_settings->gpu_allocation_ratio <= 1.0)
                {
                    cpu_allocation = (int)(processor_wall_point_count*(1.0 - user_settings->gpu_allocation_ratio));
                }
                else
                {
                    cpu_allocation = processor_wall_point_count;
                    print_warning<const char*>("Invalid user specified wall model GPU allocation ratio. Defaulting to 0.0");
                }
                break;
            }
            case WM_ESTIMATE_OCCUPANCY_BOUND:
            {
                //This too is very tricky
                break;
            }
            case WM_ALL_GPU:
            {
                cpu_allocation = 0;
                break;
            }
            case WM_NO_GPU:
            {
                gpu_allocation = 0;
                break;
            }
        }
        gpu_allocation = processor_wall_point_count - cpu_allocation;
		MPI_Allreduce(&gpu_allocation, &node_total_gpu_allocation, 1, MPI_INT, MPI_SUM, comm->node_comm);
#endif
    }


    void GlobalMemHandler::get_gpu_mem_offset(void)
    {
        int* node_gpu_point_counts = new int[comm->node_proc_info.process_count];
        node_gpu_point_counts[comm->node_proc_info.process_id] = gpu_allocation;
        MPI_Allgather(node_gpu_point_counts + comm->node_proc_info.process_id, 1, MPI_INT, node_gpu_point_counts, 1, MPI_INT, comm->node_comm);
        for (int i = 0; i < comm->node_proc_info.process_id; i++)
        {
            gpu_io_offset += *(node_gpu_point_counts+i);
        }
        delete[] node_gpu_point_counts;
    }


#if(CUDA_ENABLE)
    template void GlobalMemHandler::data_transfer<__dev_real>(cudaMemcpyKind);
    template <typename my_type>
    void GlobalMemHandler::data_transfer(cudaMemcpyKind transfer_protocol)
    {
        for (int i = 0; i < source_count; i++)
        {
            if (mem_copy_dirs[i] == transfer_protocol)
            {
                if (!(transfer_single_only[i]) || (transfer_single_only[i] && !(already_transferred[i])) && !no_gpu_buffer[i])
                {
                    already_transferred[i] = true;
                    CU_ERROR_CHECK_GMH(cudaMemcpy(
                        (my_type*)(*(transfer_endpoints[i])) + transfer_stride[i]*transfer_endpoint_offsets[i],
                        (my_type*)(*(transfer_sources[i])) + transfer_stride[i]*transfer_source_offsets[i],
                        transfer_size[i],
                        transfer_protocol));
                }
            }
        }
    }
#endif


    void GlobalMemHandler::output_endpoint_info(void)
    {
#if(CUDA_ENABLE)
        if (!(comm->is_node_root_process) && user_settings->verbose && (gpu_allocation > 0))
        {
            for (int i = 0; i < endpoint_count; i++)
            {
                std::cout << "[I] Endpoint:" << std::endl;
                std::cout << "      |" << std::endl;
                std::cout << "      | Source variable:   " << transfer_source_names[i] << " (" << *(transfer_sources[i]) << ")" << std::endl;
                std::cout << "      | Source hardware:   " << (target_is_device_pointer[i] ? "host" : "device")<< std::endl;
                std::cout << "      | Source stride:     " << transfer_stride[i] << std::endl;
                std::cout << "      | Source offset:     " << transfer_source_offsets[i] << std::endl;
                std::cout << "      |" << std::endl;
                std::cout << "      | Target variable:   " << transfer_endpoint_names[i] << " (" << *(transfer_endpoints[i]) << ")" << std::endl;
                std::cout << "      | Target hardware:   " << (!target_is_device_pointer[i] ? "host" : "device")<< std::endl;
                std::cout << "      | Target stride:     " << transfer_stride[i] << std::endl;
                std::cout << "      | Target offset:     " << transfer_endpoint_offsets[i] << std::endl;
                std::cout << "      |" << std::endl;
                std::cout << "      | Transfer protocol: " << ( (mem_copy_dirs[i] == cudaMemcpyDeviceToHost) ? "cudaMemcpyDeviceToHost" : "cudaMemcpyHostToDevice") << std::endl;
                std::cout << "      | Transfer size:     " << transfer_size[i] << std::endl;
                std::cout << "      |" << std::endl;
            }
        }
#endif
    }







    template <typename __buftype>
    void GlobalMemHandler::add_managed(const char* var_name, Buffer<__buftype*>* managed_buffer, int _stride, int _num_instance_per_ray_point, int manage_mode)
    {
        bool on_gpu = getflag(manage_mode,MANAGED_GPU);
        if (getflag(manage_mode, RESTART_BUFFER) && _stride > 1)
        {
            ASSERTKILL("RESTART_BUFFER specified for solution buffer variable \"" << var_name << "\" with dimension > 1; this is not supported (yet).");
        }
        int buffer_wall_points = 0;
        int buffer_transfer_count = 0;
        if (on_gpu)
        {
            buffer_wall_points = node_total_gpu_allocation;
            buffer_transfer_count = gpu_allocation;
        }
        else
        {
            buffer_wall_points = cpu_allocation;
            buffer_transfer_count = gpu_allocation;
        }

        managed_buffer->stride = _stride;
        managed_buffer->size = _stride * buffer_wall_points * _num_instance_per_ray_point * sizeof(__buftype);

        if (user_settings->verbose && comm->is_node_root_process && ((int)(managed_buffer->size) > 0))
        {
            std::cout << "[I] Wall model variable \"" << var_name << "\": " << (on_gpu?"GPU":"CPU") << " buffer, " << (int)(managed_buffer->size) << " bytes." << std::endl;
        }


        if (on_gpu) add_gpu_managed<__buftype>(var_name, managed_buffer, manage_mode);
        else add_cpu_managed<__buftype>(var_name, managed_buffer, manage_mode);

        if (CUDA_ENABLE) add_endpoint_if_necessary<__buftype>(var_name, managed_buffer, buffer_transfer_count, manage_mode);
    }

    template void GlobalMemHandler::add_endpoint_if_necessary<__dev_real>(const char*, Buffer<__dev_real*>*, int, int);
    template <typename __buftype>
    void GlobalMemHandler::add_endpoint_if_necessary(const char* varname, Buffer<__buftype*>* managed_buffer, int buffer_transfer_count, int manage_mode)
    {
        bool is_endpoint = getflag(manage_mode, CPU_ENDPOINT) || getflag(manage_mode, GPU_ENDPOINT) || getflag(manage_mode, GPU_ONETIME_ENDPOINT);
        bool gpu_target = (getflag(manage_mode, GPU_ENDPOINT) || getflag(manage_mode, GPU_ONETIME_ENDPOINT));
        bool gpu_pointer = getflag(manage_mode, MANAGED_GPU);
        bool is_single_copy = getflag(manage_mode, GPU_ONETIME_ENDPOINT);
        bool is_source = gpu_pointer != gpu_target;
        //
        //                   gpu pointer         cpu pointer
        //
        //     gpu target    not source          source
        //
        //
        //     cpu target    source              not source
        //

        if (is_endpoint)
        {
            if (is_source)
            {
                transfer_sources[source_endpoint_table_index] = (void**)&(managed_buffer->base);
                transfer_single_only[source_endpoint_table_index] = is_single_copy;
                already_transferred[source_endpoint_table_index] = false;
                transfer_size[source_endpoint_table_index] = buffer_transfer_count *(managed_buffer->stride) * sizeof(__buftype);
                transfer_endpoint_avail[source_endpoint_table_index] = managed_buffer->size;
                transfer_stride[source_endpoint_table_index] = managed_buffer->stride;
#if(CUDA_ENABLE)
                mem_copy_dirs[source_endpoint_table_index] = gpu_target ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
                if (user_settings->verbose && comm->is_node_root_process && managed_buffer->size > 0) std::cout << "    ------>>> " << (gpu_pointer ? "device" : "host") << " source created";
                if (user_settings->verbose && comm->is_node_root_process && managed_buffer->size > 0) std::cout << (is_single_copy ? " (single use)" : "") << std::endl;
#endif
                transfer_source_offsets[source_endpoint_table_index] = gpu_target ? cpu_allocation : gpu_io_offset;
                transfer_source_names[source_endpoint_table_index] = varname;
                source_count++;
                source_endpoint_table_index++;
            }
            else
            {
                transfer_endpoints[source_endpoint_table_index] = (void**)&(managed_buffer->base);
                target_is_device_pointer[source_endpoint_table_index] = gpu_target;
                transfer_source_avail[source_endpoint_table_index] = managed_buffer->size;
                transfer_endpoint_offsets[source_endpoint_table_index] = gpu_target ? gpu_io_offset : cpu_allocation;
                transfer_endpoint_names[source_endpoint_table_index] = varname;
#if(CUDA_ENABLE)
                if (user_settings->verbose && comm->is_node_root_process && managed_buffer->size > 0) std::cout << "    ------>>> " << (gpu_pointer ? "device" : "host") << " endpoint created";
                if (user_settings->verbose && comm->is_node_root_process && managed_buffer->size > 0) std::cout << (is_single_copy ? " (single use)" : "") << std::endl;
#endif
                endpoint_count++;
                source_endpoint_table_index++;
            }
        }
    }


    template <typename __buftype>
    void GlobalMemHandler::add_cpu_managed(const char* varname, Buffer<__buftype*>* managed_buffer, int manage_mode)
    {


        cpu_managed[cpu_managed_count] = (void**)&(managed_buffer->base);
        cpu_manage_modes[cpu_managed_count] = manage_mode;
        cpu_managed_sizes[cpu_managed_count] = managed_buffer-> size;
        cpu_var_names[cpu_managed_count] = varname;
        is_restart_buffer[cpu_managed_count] = getflag(manage_mode, RESTART_BUFFER);
        if (is_restart_buffer[cpu_managed_count])
        {
            cpu_restart[num_cpu_restart] = (void**)&(managed_buffer->base);
            num_cpu_restart++;
        }

        //HERE
        double* temporary_buffer;
        bool user_has_provided_buffer = user_variables->Associated(varname, &temporary_buffer);
        if (user_has_provided_buffer) *(cpu_managed[cpu_managed_count]) = *((void**)(&temporary_buffer));
        if (!user_has_provided_buffer && getflag(manage_mode, USER_MUST_PROVIDE))
        {
            ASSERTKILL("user required to provide buffer association for variable \"" << varname << "\", but none provided.");
        }
        bool user_can_provide_but_did_not = (getflag(manage_mode, USER_CAN_PROVIDE) && !user_has_provided_buffer);
        bool cpu_allocate_this_one = getflag(manage_mode, SOLUTION_BUFFER) || user_can_provide_but_did_not;
        cpu_required_to_allocate[cpu_managed_count] = cpu_allocate_this_one;
        cpu_already_freed[cpu_managed_count] = false;

        provide_variable_association[cpu_managed_count] = user_can_provide_but_did_not;

        cpu_managed_count++;

    }



    template <typename __buftype>
    void GlobalMemHandler::add_gpu_managed(const char* varname, Buffer<__buftype*>* managed_buffer, int manage_mode)
    {
        gpu_managed[gpu_managed_count] = (void**)&(managed_buffer->base);
        gpu_manage_modes[gpu_managed_count] = manage_mode;
        gpu_managed_sizes[gpu_managed_count] = managed_buffer-> size;
        gpu_var_names[gpu_managed_count] = varname;
        gpu_already_freed[gpu_managed_count] = false;
        no_gpu_buffer[gpu_managed_count] = getflag(manage_mode, CPU_ONLY);
        gpu_managed_count++;
    }


    template void GlobalMemHandler::allocate_buffers<__dev_real>(void);
    template <typename __buftype>
    void GlobalMemHandler::allocate_buffers(void)
    {
        if (cpu_allocation > 0)
        {
            for (int i = 0; i < cpu_managed_count; i++)
            {
                if (cpu_required_to_allocate[i]) *(cpu_managed[i]) = (__buftype*) malloc(cpu_managed_sizes[i]);
                if (provide_variable_association[i])
                {
                    user_variables->AssociateVariable(cpu_var_names[i], (double*)(*(cpu_managed[i])));
                }
            }
        }
#if(CUDA_ENABLE)
        if (node_total_gpu_allocation > 0)
        {
            if (comm->is_node_root_process)
            {
                for (int i = 0; i < gpu_managed_count; i++)
                {
                    if (!no_gpu_buffer[i])
                    {
                        CU_ERROR_CHECK_GMH(cudaMalloc((void**)gpu_managed[i], gpu_managed_sizes[i]));
#if(WM_DEBUG_OUTPUT)
                        std::cout << "Allocated device pointer: \"" << gpu_var_names[i] << "\": size=" << gpu_managed_sizes[i] << ", address=" << (void**)gpu_managed[i] << std::endl;
#endif
                        CU_ERROR_CHECK_GMH(cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&gpu_handles[i], *(gpu_managed[i])));
                    }
                }
            }
            for (int i = 0; i < gpu_managed_count; i++) MPI_Bcast(gpu_handles[i].reserved, 64, MPI_CHAR, 0, comm->node_comm);
            if (!(comm->is_node_root_process))
            {
                for (int i = 0; i < gpu_managed_count; i++)
                {
                    CU_ERROR_CHECK_GMH(cudaIpcOpenMemHandle(gpu_managed[i], gpu_handles[i], cudaIpcMemLazyEnablePeerAccess));
                }
            }
        }
#endif
    }



    void GlobalMemHandler::free_buffers(void)
    {
        if (cpu_allocation > 0)
        {
            for (int i = 0; i < cpu_managed_count; i++)
            {
                if (cpu_required_to_allocate[i] && !cpu_already_freed[i])
                {
                    free(*(cpu_managed[i]));
                }
            }
        }
#if(CUDA_ENABLE)
        if (node_total_gpu_allocation > 0)
        {
            if (comm->is_node_root_process)
            {
                for (int i = 0; i < gpu_managed_count; i++)
                {
                    if (!gpu_already_freed[i] && !no_gpu_buffer[i])
                    {
                        CU_ERROR_CHECK_GMH(cudaFree(*(gpu_managed[i])));
                    }
                }
            }
            else
            {
                for (int i = 0; i < gpu_managed_count; i++)
                {
                    CU_ERROR_CHECK_GMH(cudaIpcCloseMemHandle(*(gpu_managed[i])));
                }
            }
        }
#endif
    }


    template void GlobalMemHandler::add_RestrictedBufferGroup<__dev_real>(RestrictedBufferGroup<__dev_real*>*, const int, const int);
    template <typename __buftype>
    void GlobalMemHandler::add_RestrictedBufferGroup(RestrictedBufferGroup<__buftype*>* managed_buffer, const int wall_model_selection, const int manage_mode)
    {
        int dim2 = (2+iins3d)*(2+iins3d);
        int dim = (2+iins3d);
        int ray_num = user_settings->ray_point_count;

        if (user_settings->enable_transition_sensor)
        {
            //Wall model can't compute the gradients at the ray tip.
            add_managed<double>("strain_rate",     &(managed_buffer->in.strain_rate),     1, 1, manage_mode | USER_MUST_PROVIDE);

            add_managed<double>("strain_rate_avg", &(managed_buffer->in.strain_rate_avg), 1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("k",               &(managed_buffer->in.k),               1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("k_avg",           &(managed_buffer->in.k_avg),           1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("rho_avg",         &(managed_buffer->in.rho_avg),         1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("mu_avg",          &(managed_buffer->in.mu_avg),          1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("u_avg",           &(managed_buffer->in.u_avg),           1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("u_sq_avg",        &(managed_buffer->in.u_sq_avg),        1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("v_avg",           &(managed_buffer->in.v_avg),           1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("v_sq_avg",        &(managed_buffer->in.v_sq_avg),        1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("w_avg",           &(managed_buffer->in.w_avg),           1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("w_sq_avg",        &(managed_buffer->in.w_sq_avg),        1, 1, manage_mode | USER_CAN_PROVIDE | CPU_ONLY);
            add_managed<double>("sensor_val",      &(managed_buffer->in.sensor_val),      1, 1, manage_mode | USER_CAN_PROVIDE | GPU_ENDPOINT);
        }

        switch(user_settings->model_selection)
        {
            default:
            case MODEL_ALGEBRAIC:
            {
                if (user_settings->include_energy_equation)
                {
                    ASSERTKILL("Energy equation is not implemented for algebraic model. Please set \"solve_energy_equation\" to \"false\".");
                }
                //Usage:            name          pointer                   dimension     num/ray         management protocol flags

                //inputs
                add_managed<double>("distance",    &(managed_buffer->in.distance),     1,    1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("metric_data", &(managed_buffer->in.metric_data),  dim2, 1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("x",           &(managed_buffer->in.x),            1,    1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("coord_sys",   &(managed_buffer->in.coord_sys),    dim2, 1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("p",           &(managed_buffer->in.p),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("u",           &(managed_buffer->in.u),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("v",           &(managed_buffer->in.v),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("w",           &(managed_buffer->in.w),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("T",           &(managed_buffer->in.T),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("turb",        &(managed_buffer->in.turb),         1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("rho",         &(managed_buffer->in.rho),          1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("mu_lam",      &(managed_buffer->in.mu_lam),       1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);

                //solution
                add_managed<__buftype>("u",     &(managed_buffer->solution.u),     1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("d",     &(managed_buffer->solution.d),     1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("nu",    &(managed_buffer->solution.nu),    1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("rho",   &(managed_buffer->solution.rho),   1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("uplus", &(managed_buffer->solution.uplus), 1, ray_num, manage_mode | SOLUTION_BUFFER);
                add_managed<__buftype>("yplus", &(managed_buffer->solution.yplus), 1, ray_num, manage_mode | SOLUTION_BUFFER);

                //outputs
                add_managed<double>("stress_tensors", &(managed_buffer->out.stress_tensors), dim2, 1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("tau",            &(managed_buffer->out.tau),            1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("vorticity",      &(managed_buffer->out.vorticity),      1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("residual",       &(managed_buffer->out.residual),       1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("iterations",     &(managed_buffer->out.iterations),     1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                break;
            }
            case MODEL_ODE_NO_PRESSURE_GRADIENT_NO_CONVECTION:
            {
                //inputs
                add_managed<double>("distance",    &(managed_buffer->in.distance),     1,    1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("metric_data", &(managed_buffer->in.metric_data),  dim2, 1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("x",           &(managed_buffer->in.x),            1,    1,      manage_mode | GPU_ONETIME_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("coord_sys",   &(managed_buffer->in.coord_sys),    dim2, 1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("p",           &(managed_buffer->in.p),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("u",           &(managed_buffer->in.u),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("v",           &(managed_buffer->in.v),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("w",           &(managed_buffer->in.w),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("T",           &(managed_buffer->in.T),            1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("turb",        &(managed_buffer->in.turb),         1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("rho",         &(managed_buffer->in.rho),          1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);
                add_managed<double>("mu_lam",      &(managed_buffer->in.mu_lam),       1,    1,      manage_mode | GPU_ENDPOINT | USER_MUST_PROVIDE);


                //solution
                add_managed<__buftype>("u",     &(managed_buffer->solution.u),     1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("d",     &(managed_buffer->solution.d),     1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("turb",  &(managed_buffer->solution.turb),  1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("rho",   &(managed_buffer->solution.rho),   1, ray_num, manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                add_managed<__buftype>("nu",    &(managed_buffer->solution.nu),    1, ray_num, manage_mode | SOLUTION_BUFFER);// | INIT_ONLY);
                add_managed<__buftype>("uplus", &(managed_buffer->solution.uplus), 1, ray_num, manage_mode | SOLUTION_BUFFER);// | INIT_ONLY);
                add_managed<__buftype>("yplus", &(managed_buffer->solution.yplus), 1, ray_num, manage_mode | SOLUTION_BUFFER);// | INIT_ONLY);

                //Left-hand side for newton iteration.
                //[dMdM dMdT]
                //[dTdM dTdT]
                //or
                //[dMdM dMdT dMdE]
                //[dTdM dTdT dTdE]
                //[dEdM dEdT dEdE]
                //sub =  subdiagonal
                //sup =  superdiagonal
                //diag = diagonal
                int num_eqs = user_settings->include_energy_equation?3:2;
                //                                                               j+i*num_eqs
                if (user_settings->include_energy_equation)
                {
                    add_managed<__buftype>("Qwall",              &(managed_buffer->out.Qwall),                1, 1,         manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                    add_managed<__buftype>("heat_flux_vector",   &(managed_buffer->out.heat_flux_vector),     dim, 1,       manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                    add_managed<__buftype>("T",                  &(managed_buffer->solution.T),               1, ray_num,   manage_mode | SOLUTION_BUFFER | RESTART_BUFFER);
                }

                //lhs
                for (int b = 0; b < num_eqs*num_eqs; b++)
                {
                    add_managed<__buftype>("odelhs1",  &(managed_buffer->solution.diags[b]),                  1, ray_num-2, manage_mode | SOLUTION_BUFFER);
                    add_managed<__buftype>("odelhs2",  &(managed_buffer->solution.subdiags[b]),               1, ray_num-3, manage_mode | SOLUTION_BUFFER);
                    add_managed<__buftype>("odelhs3",  &(managed_buffer->solution.supdiags[b]),               1, ray_num-3, manage_mode | SOLUTION_BUFFER);
                }

                //rhs
                for (int b = 0; b < num_eqs; b++)
                {
                    add_managed<__buftype>("rhs",  &(managed_buffer->solution.rhs[b]), 1, ray_num-2, manage_mode | SOLUTION_BUFFER);
                }


                //outputs
                add_managed<double>("stress_tensors", &(managed_buffer->out.stress_tensors), dim2, 1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("tau",            &(managed_buffer->out.tau),            1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("vorticity",      &(managed_buffer->out.vorticity),      1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("residual",       &(managed_buffer->out.residual),       1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                add_managed<double>("iterations",     &(managed_buffer->out.iterations),     1,    1,      manage_mode | CPU_ENDPOINT | USER_CAN_PROVIDE);
                break;
            }
            case MODEL_ODE_PRESSURE_GRADIENT_CONVECTION:
            {
                std::cout << "WMLIB MODEL_ODE_PRESSURE_GRADIENT_CONVECTION NOT IMPLEMENTED!!!";
                WAIT;
                break;
            }
        }
        source_endpoint_table_index = 0;
    }
}
