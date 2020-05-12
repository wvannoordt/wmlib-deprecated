#include "wall_model_worker.h"
#include "wm_lib_typedef.h"
#include "WmTestData.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
namespace wall_model_module
{
    bool WmTestData::fileexists(const char* name)
    {
        std::string namestr(name);
        std::ifstream f(namestr.c_str());
        return f.good();
    }
    WmTestData::WmTestData(InputBuffers input_in, OutputBuffers output_in, int npts_in, UserSettings settings_in)
    {
        input = input_in;
        output = output_in;
        npts = npts_in;
        num_bufs = 0;
        has_external_handling = true;
        require_free = false;
        settings = settings_in;
        populate_relevant_buffers(settings.model_selection);
    }

    double* WmTestData::operator [] (const char* varname)
    {
        double* output;
        if(!find_variable(varname, &output))
        {
            std::cout << "Error: cannot find variable \"" << varname << "\". Stopping." << std::endl;
            throw "(fatal exception)";
        }
        return output;
    }

    WmTestData::WmTestData(void)
    {
        npts = 0;
        num_bufs = 0;
        has_external_handling = false;
        require_free = false;
    }
    WmTestData::~WmTestData(void)
    {
        close();
    }

    bool WmTestData::find_variable(const char* varname, double** var_ptr_out)
    {
        int hash = get_hash_id(varname);
        for (int i = 0; i < num_bufs; i++)
        {
            if (name_hashes[i] == hash)
            {
                *var_ptr_out = relevant_bufs[i]->base;
                return true;
            }
        }
        return false;
    }

    int WmTestData::get_hash_id(const char* s)
    {
        int h = CONST_HASH_H;
        while (*s)
        {
            h = (h * CONST_HASH_A) ^ (s[0] * CONST_HASH_B);
            s++;
        }
        return h;
    }

    void WmTestData::write_to_file(const char* filename)
    {
        FILE* file_writer;
        file_writer = fopen(filename, "w+b");
        settings.write_serialized(file_writer);
        fwrite(&num_bufs, sizeof(int), 1, file_writer);
        fwrite(&npts, sizeof(int), 1, file_writer);
        for (int i = 0; i < num_bufs; i++)
        {
            int name_len = strlen(buf_names + i*MAX_BUFFER_NAME_LEN)+1;
            fwrite(&name_len, sizeof(int), 1, file_writer);
            fwrite(buf_names+i*MAX_BUFFER_NAME_LEN, sizeof(char), name_len, file_writer);
            fwrite(relevant_bufs[i]->base, sizeof(double), npts, file_writer);
        }
        fclose(file_writer);
    }

    void WmTestData::read_from_file(const char* filename)
    {
        if (!fileexists(filename))
        {
            CPUERR("Cannot find file " << filename);
        }

        FILE* file_writer;
        file_writer = fopen(filename, "r+b");
        settings.read_serialized(file_writer);
        populate_relevant_buffers(settings.model_selection);
        fread(&num_bufs, sizeof(int), 1, file_writer);
        fread(&npts, sizeof(int), 1, file_writer);
        allocateall();
        for (int i = 0; i < num_bufs; i++)
        {
            int num_chars_in_name;
            fread(&num_chars_in_name, sizeof(int), 1, file_writer);
            fread(buf_names+i*MAX_BUFFER_NAME_LEN, sizeof(char), num_chars_in_name, file_writer);
            name_hashes[i] = get_hash_id(buf_names+i*MAX_BUFFER_NAME_LEN);
            fread(relevant_bufs[i]->base, sizeof(double), npts, file_writer);
        }
        fclose(file_writer);
    }

    void WmTestData::allocateall(void)
    {
        if (!require_free)
        {
            for (int i = 0; i < num_bufs; i++)
            {
                size_t bufsize = sizeof(double)*npts;
                relevant_bufs[i]->base = (double*)malloc(bufsize);
                relevant_bufs[i]->size = bufsize;
            }
            require_free = true;
        }
    }

    void WmTestData::close(void)
    {
        if (!has_external_handling&&require_free) freeall();
        require_free = false;
    }

    void WmTestData::freeall(void)
    {
        for (int i = 0; i < num_bufs; i++)
        {
            free(relevant_bufs[i]->base);
        }
    }

    void WmTestData::add_buffer(Buffer<double*>* buf, const char* name)
    {
        relevant_bufs[num_bufs] = buf;
        memcpy(buf_names + num_bufs*MAX_BUFFER_NAME_LEN, name, strlen(name));
        name_hashes[num_bufs] = get_hash_id(buf_names+num_bufs*MAX_BUFFER_NAME_LEN);
        num_bufs ++;
    }

    void WmTestData::populate_relevant_buffers(int model_in)
    {
        switch (model_in)
        {
            default:
            case MODEL_ALGEBRAIC:
            {
                add_buffer(&input.x, "x-coordinate");
                add_buffer(&input.coord_sys, "coord_sys");
                add_buffer(&input.metric_data, "metric_data");
                add_buffer(&input.distance, "distance");
                add_buffer(&input.p, "pressure");
                add_buffer(&input.u, "u-velocity");
                add_buffer(&input.v, "v-velocity");
                add_buffer(&input.w, "w-velocity");
                add_buffer(&input.T, "Temperature");
                add_buffer(&input.turb, "nu_tilde");
                add_buffer(&input.rho, "density");
                add_buffer(&input.mu_lam, "laminar_viscosity");
                add_buffer(&output.tau, "wall_shear_stress");
                break;
            }
            case MODEL_ODE_NO_PRESSURE_GRADIENT_NO_CONVECTION:
            {
                //inputs
                add_buffer(&input.x, "x-coordinate");
                add_buffer(&input.coord_sys, "coord_sys");
                add_buffer(&input.metric_data, "metric_data");
                add_buffer(&input.distance, "distance");
                add_buffer(&input.p, "pressure");
                add_buffer(&input.u, "u-velocity");
                add_buffer(&input.v, "v-velocity");
                add_buffer(&input.w, "w-velocity");
                add_buffer(&input.T, "Temperature");
                add_buffer(&input.turb, "nu_tilde");
                add_buffer(&input.rho, "density");
                add_buffer(&input.mu_lam, "laminar_viscosity");
                add_buffer(&output.tau, "wall_shear_stress");
                if (settings.include_energy_equation &&  settings.adiabatic_wall) add_buffer(&output.Qwall, "wall_temperature");
                if (settings.include_energy_equation && !settings.adiabatic_wall) add_buffer(&output.Qwall, "wall_heat_flux");
                break;
            }
            case MODEL_ODE_PRESSURE_GRADIENT_CONVECTION:
            {
                std::cout << "WMLIB MODEL_ODE_PRESSURE_GRADIENT_CONVECTION NOT IMPLEMENTED! See " << __FILE__ << "::" << __LINE__ << std::endl;
                WAIT;
                break;
            }
        }
    }
}
