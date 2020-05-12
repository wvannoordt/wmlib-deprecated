#include "wm_lib_typedef.h"
#include "crc32.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace wall_model_module
{
    class pngwriter
    {
        public:
            void write_png(const char* filename_png_ext, const int& width, const int& height, double* r, double* g, double* b);
            pngwriter(void);
        private:
            bool machine_big_endian;
            crc32 crc32_checksum;
            void write_signature(FILE* file_writer);
            void write_header(FILE* file_writer, const int& width, const int& height);
            void write_data(FILE* file_writer, const int& col, const int& row, double* r_data, double* g_data, double* b_data);
            void write_end(FILE* file_writer);
            void flip_endianness(int* i);
#if(WM_ALLOW_DEBUG_EXT)
            void compress_memory(void *in_data, size_t in_data_size, std::vector<uint8_t> &out_data);
#endif
    };
}
