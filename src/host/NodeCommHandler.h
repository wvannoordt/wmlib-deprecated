#ifndef NODE_COMM_H
#define NODE_COMM_H

#include <iostream>
#include "mpi.h"
#include "wall_model_options.h"
#include "wm_lib_typedef.h"
namespace wall_model_module
{
    class NodeCommHandler
    {
        public:
            MPI_Comm global_comm;
            MPI_Comm node_comm;
            MPI_Comm gpualloc_comm;
            MPI_Comm wallpoint_comm;
            mpi_comm_info global_proc_info;
            mpi_comm_info node_proc_info;
            mpi_comm_info gpualloc_proc_info;
            mpi_comm_info wallpoint_proc_info;
            char node_name[MPI_MAX_PROCESSOR_NAME];
        	int name_len;
        	int processor_node_id;
        	int current_node_root_process_id;
        	int* proc_hash_ids;
            bool is_global_root_process;
            bool is_node_root_process;
            bool node_has_nvgpu;
            int node_nvgpu_count;

            NodeCommHandler(MPI_Comm _host_comm);
            NodeCommHandler(void);
            ~NodeCommHandler(void);
            void Close(void);
            void build_gpualloc_comm(int gpu_group_id);
            void build_wallpoint_comm(int group_id);

        private:
            bool deallocate_pointers;

            void build_node_comm(void);
    		int get_hash_id(const char* s);
            bool determine_is_node_root_process(void);
    };
    extern std::ostream & operator<<(std::ostream & os_stream, const NodeCommHandler & input);
}

#endif
