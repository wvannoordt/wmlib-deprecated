#ifndef WM_IO_H
#define WM_IO_H
#include "wall_model_worker.h"
#include "hybrid_computing.h"
#include "indexing.h"
#include "NodeCommHandler.h"
namespace wall_model_module
{
    void output_u_turb(const char* filename, HybridComputeInstance<double*> globaldata, int buffer_index);
    void output_u_turb_T(const char* filename, HybridComputeInstance<double*> globaldata, int buffer_index);
    void output_distance(const char* filename, HybridComputeInstance<double*> globaldata, int buffer_index);
    void output_u_turb_rhs(const char* filename, HybridComputeInstance<double*> globaldata, int buffer_index);
    void output_momturb_lhs(const char* filename, LINSYS_block_tridiag<__buffertype*, 2>* linear_system, int buffer_index);
    void output_rhs(const char* filename, LINSYS_block_tridiag<__buffertype*, 2>* linear_system, HybridComputeInstance<__buffertype*> globaldata, int buffer_index);
    template <const int NUM_EQ> void output_rhs_onecolumn(const char* filename, LINSYS_block_tridiag<__buffertype*, NUM_EQ>* linear_system, HybridComputeInstance<__buffertype*> globaldata, int buffer_index);
    void write_restart_file_internal(char* filename, GlobalMemHandler* mem_handler);
    void read_restart_file_internal(char* filename, GlobalMemHandler* mem_handler);
    void process_restart_file_internal(char* filename, GlobalMemHandler* mem_handler, bool read_mode);
}
#endif
