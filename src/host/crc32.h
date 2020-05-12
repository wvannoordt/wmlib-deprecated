#include "wm_lib_typedef.h"

#ifndef CRC32_H
#define CRC32_H
#define CRC32_TABLE_CT 256

namespace wall_model_module
{
    class crc32
    {
        public:
            crc32(void);
            unsigned int compute_checksum(ubyte* bytes, int length);
        private:
            unsigned int table[CRC32_TABLE_CT];
    };

}
#endif
