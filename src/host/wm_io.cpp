#include "wall_model_worker.h"
#include "hybrid_computing.h"
#include "indexing.h"
#include "GlobalMemHandler.h"
#include "NodeCommHandler.h"
#include <iostream>
#include <fstream>
#include "debug_output.h"
#include "wm_io.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
using namespace wall_model_module;
namespace HYBRID
{
    bool file_exists(char* name)
    {
        std::string namestr(name);
        std::ifstream f(namestr.c_str());
        return f.good();
    }

    void output_u_turb(const char* filename, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        for (int i = 0; i < WM_NRAY; i++) myfile << BUFFERVARIDX(d, i) << "," << BUFFERVARIDX(u,i) << "," << BUFFERVARIDX(turb,i) << "\n";
        myfile.close();
    }

    void output_u_turb_T(const char* filename, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        for (int i = 0; i < WM_NRAY; i++) myfile << BUFFERVARIDX(d, i) << "," << BUFFERVARIDX(u,i) << "," << BUFFERVARIDX(turb,i) << "," << BUFFERVARIDX(T,i) << "\n";
        myfile.close();
    }

    void output_distance(const char* filename, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        for (int i = 0; i < WM_NRAY; i++) myfile << BUFFERVARIDX(d, i) << "\n";
        myfile.close();
    }


    template <const int NUM_EQ> void output_rhs_onecolumn(const char* filename, LINSYS_block_tridiag<__buffertype*, NUM_EQ>* linear_system, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        for (int k = 0; k < NUM_EQ; k++)
        {
            for (int i = 0; i < linear_system->blockdim; i++)
                myfile << linear_system->rhs[k].base[i] << std::endl;
        }
        myfile.close();
    }
    template void output_rhs_onecolumn<2>(const char*, LINSYS_block_tridiag<__buffertype*, 2>*, int);
    template void output_rhs_onecolumn<3>(const char*, LINSYS_block_tridiag<__buffertype*, 3>*, int);

    void output_rhs(const char* filename, LINSYS_block_tridiag<__buffertype*, 2>* linear_system, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        for (int k = 0; k < 2; k++)
        {
            for (int i = 0; i < linear_system->blockdim; i++)
            myfile << BUFFERVARIDX(d, i+1) << ", " << linear_system->rhs[k].base[i] << std::endl;
        }
        myfile.close();
    }

    void output_u_turb_rhs(const char* filename, HybridComputeInstance<double*> globaldata, int buffer_index)
    {
        std::ofstream myfile;
        myfile.open(filename);
        double mom, turb;
        double du[3] = {0,0,0};
        double dnu[3] = {0,0,0};
        for (int idx = 0; idx < WM_NRAY-2; idx++)
        {

            compute_rhs_turb_mom<__hybrid>(buffer_index, &mom, &turb, du, dnu, idx+1, 1.0);
            myfile << BUFFERVARIDX(d, idx+1) << "," << mom << "," << turb << "\n";
        }

        myfile.close();
    }
}
