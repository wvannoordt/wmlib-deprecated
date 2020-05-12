#ifndef WM_TEST_DAT_H
#define WM_TEST_DAT_H

#include "wm_lib_typedef.h"
#include <string.h>
namespace wall_model_module
{
    class WmTestData
    {
        public:
            WmTestData(InputBuffers input_in, OutputBuffers output_in, int npts_in, UserSettings settings_in);
            WmTestData(void);
            ~WmTestData(void);
            void write_to_file(const char* filename);
            void read_from_file(const char* filename);
            void close(void);
            bool find_variable(const char* varname, double** var_ptr_out);
            InputBuffers input;
            OutputBuffers output;
            int npts;
            double* operator [] (const char* varname);
            UserSettings settings;
        private:
            int get_hash_id(const char* s);
            void populate_relevant_buffers(int model_in);
            bool fileexists(const char* name);
            void allocateall(void);
            void add_buffer(Buffer<double*>* buf, const char* name);
            void freeall(void);
            Buffer<double*>* relevant_bufs[MAX_BUFFERS];
            char buf_names[MAX_BUFFERS*MAX_BUFFER_NAME_LEN] = {0};
            int name_hashes[MAX_BUFFERS];
            int num_bufs;
            bool has_external_handling, require_free;
    };
}

#endif
