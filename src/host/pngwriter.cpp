#include "pngwriter.h"
#include "crc32.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#if(WM_ALLOW_DEBUG_EXT)
#include "zlib.h"
#endif
#include <vector>
#include <assert.h>
#include <cstring>

namespace wall_model_module
{
    pngwriter::pngwriter()
    {
        crc32_checksum = crc32();
        int num = 1;
        machine_big_endian = ! ( *(char *)&num == 1 );
    }
    void pngwriter::write_png(const char* filename_png_ext, const int& width, const int& height, double* r_data, double* g_data, double* b_data)
    {
        FILE* file_writer;
        file_writer = fopen(filename_png_ext, "w+b");
        write_signature(file_writer);
        write_header(file_writer, width, height);
        write_data(file_writer, width, height, r_data, g_data, b_data);
        write_end(file_writer);
        fclose(file_writer);
    }
    void pngwriter::write_data(FILE* file_writer, const int& col, const int& row, double* r_data, double* g_data, double* b_data)
    {

        ubyte* buffer_in;
        size_t input_buffer_size = ((3*col + 1) * row) * sizeof(ubyte);

        buffer_in = (ubyte*)malloc(input_buffer_size);
        //set line filter bytes

        int i, j;
        #pragma omp simd
        for (i = 0; i < row; i++) *(buffer_in + i*(3*col + 1)) = 0x00;

        //filter r g b r g b r g b ...
        #pragma omp simd
        for (i = 0; i < row; i++)
        {
            for (j = 0; j < col; j++)
            {
                *(buffer_in + i*(3*col + 1) + 3*j + 1) = (ubyte)(*(r_data + i*col + j) * 255);
                *(buffer_in + i*(3*col + 1) + 3*j + 2) = (ubyte)(*(g_data + i*col + j) * 255);
                *(buffer_in + i*(3*col + 1) + 3*j + 3) = (ubyte)(*(b_data + i*col + j) * 255);
            }
        }

        std::vector<uint8_t> compressed_buffer_vec;
#if(WM_ALLOW_DEBUG_EXT)
        compress_memory(buffer_in, input_buffer_size, compressed_buffer_vec);
#else
        std::cout << "Error in file " << __FILE__ << ", line " << __LINE__ << ": attempted invalid call to pngwriter::compress_memory. Recompile wall model library with WM_ALLOW_DEBUG_EXT=1 and ensure you have zlib.\n";
#endif
        int comp_size = static_cast<int>(compressed_buffer_vec.size());
        int comp_size_write = comp_size;
        if (!machine_big_endian) flip_endianness(&comp_size_write);

        ubyte* comp_buffer_in = (ubyte*)malloc((4+comp_size)*sizeof(ubyte));
        std::copy(compressed_buffer_vec.begin(),compressed_buffer_vec.end(), comp_buffer_in + 4);
        *(comp_buffer_in+0) = 'I';
        *(comp_buffer_in+1) = 'D';
        *(comp_buffer_in+2) = 'A';
        *(comp_buffer_in+3) = 'T';

        fwrite(&comp_size_write, 1, sizeof(int), file_writer);
        fwrite(comp_buffer_in, 1, (4+comp_size)*sizeof(ubyte), file_writer);

        int crc32_data = crc32_checksum.compute_checksum(comp_buffer_in, 4+comp_size);
        if (!machine_big_endian) flip_endianness(&crc32_data);
        fwrite(&crc32_data, 1, sizeof(int), file_writer);
        free(comp_buffer_in);
        free(buffer_in);
    }
#if(WM_ALLOW_DEBUG_EXT)
    void pngwriter::compress_memory(void *in_data, size_t in_data_size, std::vector<uint8_t> &out_data)
    {
        std::vector<uint8_t> buffer;

        const size_t BUFSIZE = 128 * 1024;
        uint8_t temp_buffer[BUFSIZE];

        z_stream strm;
        strm.zalloc = 0;
        strm.zfree = 0;
        strm.next_in = reinterpret_cast<uint8_t *>(in_data);
        strm.avail_in = in_data_size;
        strm.next_out = temp_buffer;
        strm.avail_out = BUFSIZE;

        deflateInit(&strm, Z_BEST_COMPRESSION);

        while (strm.avail_in != 0)
        {
            int res = deflate(&strm, Z_NO_FLUSH);
            assert(res == Z_OK);
            if (strm.avail_out == 0)
            {
                buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
                strm.next_out = temp_buffer;
                strm.avail_out = BUFSIZE;
            }
        }

        int deflate_res = Z_OK;
        while (deflate_res == Z_OK)
        {
            if (strm.avail_out == 0)
            {
                buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE);
                strm.next_out = temp_buffer;
                strm.avail_out = BUFSIZE;
            }
            deflate_res = deflate(&strm, Z_FINISH);
        }

        assert(deflate_res == Z_STREAM_END);
        buffer.insert(buffer.end(), temp_buffer, temp_buffer + BUFSIZE - strm.avail_out);
        deflateEnd(&strm);
        out_data.swap(buffer);
    }
#endif

    void pngwriter::write_signature(FILE* file_writer)
    {
        ubyte png_signature[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
        fwrite(&png_signature , sizeof(ubyte), 8*sizeof(ubyte), file_writer);
    }
    void pngwriter::write_header(FILE* file_writer, const int& width, const int& height)
    {
        int header_length = 13;
        int header_length_write = 13;

        ubyte header[header_length+4];

        int output_width = width;
        int output_height = height;
        ubyte bit_depth = 0x8;
        ubyte color_type = 0x2;
        ubyte compression_method = 0x0;
        ubyte interlace_method = 0x0;
        ubyte filter_method = 0x0;
        if (!machine_big_endian) flip_endianness(&header_length_write);
        if (!machine_big_endian) flip_endianness(&output_width);
        if (!machine_big_endian) flip_endianness(&output_height);
        std::memcpy(header, "IHDR", 4);
        std::memcpy(header+4, &output_width, 4);
        std::memcpy(header+8, &output_height, 4);
        std::memcpy(header+12, &bit_depth, 1);
        std::memcpy(header+13, &color_type, 1);
        std::memcpy(header+14, &compression_method, 1);
        std::memcpy(header+15, &filter_method, 1);
        std::memcpy(header+16, &interlace_method, 1);
        fwrite(&header_length_write, 1, sizeof(int), file_writer);
        fwrite(header, 1, (4+header_length)*sizeof(ubyte), file_writer);

        int checksum = crc32_checksum.compute_checksum(header, header_length+4);
        if (!machine_big_endian) flip_endianness(&checksum);
        fwrite(&checksum, 1, sizeof(int), file_writer);
    }
    void pngwriter::write_end(FILE* file_writer)
    {
        ubyte endblock[12] = {0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
        fwrite(endblock, 1, 12*sizeof(ubyte), file_writer);
    }
    void pngwriter::flip_endianness(int* input)
    {
        int j;
        ubyte byte_value;
        for (j = 0; j < sizeof(int)/2; j++)
        {
            byte_value = *(((ubyte*)input)+j);
            *(((ubyte*)input)+j) = *(((ubyte*)input)+sizeof(int)-j-1);
            *(((ubyte*)input)+sizeof(int)-j-1) = byte_value;
        }
    }
}
