#ifndef WM_USER_PROV_ASSOC
#define WM_USER_PROV_ASSOC
#include "WmTestData.h"
#include "wm_lib_typedef.h"

namespace wall_model_module
{
    class ProvidedVariableAssociations
    {
        public:
            ProvidedVariableAssociations();
            void AssociateVariable(const char* name, double* data_location);
            void AssociateVariableFromInputData(WmTestData* data);
            bool Associated(const char* name, double** buffer_addr);
        private:
            int get_hash_id(const char* s);
            const char* buffer_names[MAX_BUFFERS] = {0};
            double* buffer_pointers[MAX_BUFFERS];
            int name_hashes[MAX_BUFFERS];
            int num_bufs;
    };
}

#endif
