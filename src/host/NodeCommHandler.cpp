#include "mpi.h"
#include "wall_model_options.h"
#include "NodeCommHandler.h"
#include "wm_lib_typedef.h"
#if (CUDA_ENABLE)
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif
#include <iostream>
namespace wall_model_module
{
    NodeCommHandler::NodeCommHandler(void)
    {
        //Empty
    }

    NodeCommHandler::NodeCommHandler(MPI_Comm _host_comm)
    {
        MPI_Comm_dup(_host_comm, &global_comm);
        MPI_Comm_rank(global_comm, &global_proc_info.process_id);
        MPI_Comm_size(global_comm, &global_proc_info.process_count);
        MPI_Get_processor_name(node_name, &name_len);

        is_global_root_process = (global_proc_info.process_id == 0);
        processor_node_id = get_hash_id(node_name);

        deallocate_pointers = true;
        proc_hash_ids = new int[global_proc_info.process_count];
        is_node_root_process = determine_is_node_root_process();
#if(CUDA_ENABLE)
        build_node_comm();
        cudaGetDeviceCount(&node_nvgpu_count);
#else
        node_nvgpu_count = 0;
#endif
        node_has_nvgpu = (node_nvgpu_count > 0);
    }

    NodeCommHandler::~NodeCommHandler(void)
    {

    }
    void NodeCommHandler::Close(void)
    {
        if (deallocate_pointers)
        {
            delete[] proc_hash_ids;
        }
    }

    void NodeCommHandler::build_node_comm(void)
    {
        MPI_Comm_split(global_comm, processor_node_id, global_proc_info.process_id, &node_comm);
        MPI_Comm_rank(node_comm, &node_proc_info.process_id);
		MPI_Comm_size(node_comm, &node_proc_info.process_count);
    }

    void NodeCommHandler::build_gpualloc_comm(int gpu_group_id)
    {
        MPI_Comm_split(node_comm, gpu_group_id, node_proc_info.process_id, &gpualloc_comm);
        MPI_Comm_rank(gpualloc_comm, &gpualloc_proc_info.process_id);
		MPI_Comm_size(gpualloc_comm, &gpualloc_proc_info.process_count);
    }

    void NodeCommHandler::build_wallpoint_comm(int group_id)
    {
        MPI_Comm_split(global_comm, group_id, node_proc_info.process_id, &wallpoint_comm);
        MPI_Comm_rank(wallpoint_comm, &wallpoint_proc_info.process_id);
		MPI_Comm_size(wallpoint_comm, &wallpoint_proc_info.process_count);
    }

    bool NodeCommHandler::determine_is_node_root_process(void)
    {
        proc_hash_ids[global_proc_info.process_id] = processor_node_id;
        MPI_Allgather(&(global_proc_info.process_id), 1, MPI_INT, proc_hash_ids, 1, MPI_INT, global_comm);
        current_node_root_process_id = 0x00ffff;

        for (int i = 0; i < global_proc_info.process_count; i++)
        {
            if (proc_hash_ids[i] == processor_node_id)
            {
                current_node_root_process_id = (i < current_node_root_process_id) ? i : current_node_root_process_id;
            }
        }

        return (current_node_root_process_id == global_proc_info.process_id);
    }

    std::ostream & operator<<(std::ostream & os_stream, const NodeCommHandler & input)
    {
        //CURRENTLY SEGFAULTS
        os_stream << "Node info: " << input.node_name << " (" << std::hex << input.processor_node_id << "): Root process " <<  input.current_node_root_process_id << std::dec;
        return os_stream;
    }

    int NodeCommHandler::get_hash_id(const char* s)
    {
        int h = CONST_HASH_H;
        while (*s)
        {
            h = (h * CONST_HASH_A) ^ (s[0] * CONST_HASH_B);
            s++;
        }
        return h;
    }
}
