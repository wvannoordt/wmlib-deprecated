#ifndef GLOB_MEM_HAN_H
#define GLOB_MEM_HAN_H
#include "wall_model_options.h"
#include "wm_lib_typedef.h"
#include "NodeCommHandler.h"
#include "ProvidedVariableAssociations.h"
#include "mpi.h"
#include <string>
#if(CUDA_ENABLE)
#include "driver_types.h"
#endif

namespace wall_model_module
{
    class GlobalMemHandler
    {
        //CPU buffers:
        // do nothing with the input / output buffers as the main code is expected to manage those.
        // malloc and free the working buffers
        public:
            GlobalMemHandler(NodeCommHandler* _comm, int _processor_wall_point_count, UserSettings* _user_settings, ProvidedVariableAssociations* user_variables);
            GlobalMemHandler();

            bool GetNextRestartBuffer(double** buffer_output);
            void ResetRestartBufferCounter(void);

            template <typename __buftype>
            void add_managed(const char* var_name, Buffer<__buftype*>* managed_buffer, int _stride, int _num_instance_per_ray_point, int manage_mode);

            template <typename __buftype>
            void add_RestrictedBufferGroup(RestrictedBufferGroup<__buftype*>* managed_buffer, const int wall_model_selection, const int manage_mode);

            template <typename __buftype>
            void add_endpoint_if_necessary(const char* varname, Buffer<__buftype*>* managed_buffer, int buffer_transfer_count, int manage_mode);

            template <typename __buftype>
            void allocate_buffers(void);
            void free_buffers(void);
            void free_inits_buffers(void);

            void output_endpoint_info(void);

#if(CUDA_ENABLE)
            template <typename my_type>
            void data_transfer(cudaMemcpyKind transfer_protocol);
#endif
#if(CUDA_ENABLE)
            void CudaKillIfError(cudaError _cu_error, const char* file, const int line, NodeCommHandler* node_info);
#endif
            int node_total_gpu_allocation;
            int cpu_allocation;
            int gpu_allocation;
            int gpu_io_offset;
            bool inits_have_been_freed;
            int processor_wall_point_count;
            int cpu_count_before;
            int gpu_count_before;
            int total_count_before;

            UserSettings* user_settings;
            NodeCommHandler* comm;

        private:
            ProvidedVariableAssociations* user_variables;


            int cpu_manage_modes[MAX_BUFFERS];
            void** cpu_managed[MAX_BUFFERS];
            void** cpu_restart[MAX_BUFFERS];
            bool cpu_required_to_allocate[MAX_BUFFERS];
            bool cpu_already_freed[MAX_BUFFERS];
            bool is_restart_buffer[MAX_BUFFERS];
            bool provide_variable_association[MAX_BUFFERS];

            int gpu_manage_modes[MAX_BUFFERS];
            void** gpu_managed[MAX_BUFFERS];
            bool gpu_already_freed[MAX_BUFFERS];
            bool no_gpu_buffer[MAX_BUFFERS];

            void** transfer_endpoints[MAX_BUFFERS];
            void** transfer_sources[MAX_BUFFERS];

            int transfer_endpoint_offsets[MAX_BUFFERS];
            int transfer_source_offsets[MAX_BUFFERS];

            size_t transfer_endpoint_avail[MAX_BUFFERS];
            size_t transfer_source_avail[MAX_BUFFERS];

            bool   transfer_single_only[MAX_BUFFERS];
            bool   already_transferred[MAX_BUFFERS];
            size_t transfer_size[MAX_BUFFERS];

            int    transfer_stride[MAX_BUFFERS];
            bool   target_is_device_pointer[MAX_BUFFERS];

            const char* cpu_var_names[MAX_BUFFERS];
            const char* gpu_var_names[MAX_BUFFERS];
            const char* transfer_endpoint_names[MAX_BUFFERS];
            const char* transfer_source_names[MAX_BUFFERS];


            size_t cpu_managed_sizes[MAX_BUFFERS];
            size_t gpu_managed_sizes[MAX_BUFFERS];

            int cpu_managed_count;
            int gpu_managed_count;
            int source_endpoint_table_index;

            int endpoint_count;
            int source_count;

            int restart_buffer_index;
            int num_cpu_restart;




            template <typename __buftype>
            void add_cpu_managed(const char* varname, Buffer<__buftype*>* managed_buffer, int manage_mode);
            template <typename __buftype>
            void add_gpu_managed(const char* varname, Buffer<__buftype*>* managed_buffer, int manage_mode);

            template <typename my_type>
            void print_warning(my_type info);

            bool getflag(const int a, const int b);

            template <typename my_type>
            void print_info(my_type info);

            void get_allocation(void);
            void get_gpu_mem_offset(void);
#if(CUDA_ENABLE)
            cudaIpcMemHandle_t gpu_handles[MAX_BUFFERS];
            cudaMemcpyKind mem_copy_dirs[MAX_BUFFERS];
#endif
    };
}
#endif
