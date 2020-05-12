#include "wall_model_worker.h"
#include "wm_lib_typedef.h"
#include "ProvidedVariableAssociations.h"
#include "WmTestData.h"
namespace wall_model_module
{
    ProvidedVariableAssociations::ProvidedVariableAssociations(void)
    {
        num_bufs = 0;
    }

    void ProvidedVariableAssociations::AssociateVariable(const char* name, double* data_location)
    {
        buffer_names[num_bufs] = name;
        buffer_pointers[num_bufs] = data_location;
        name_hashes[num_bufs] = get_hash_id(name);
        num_bufs++;
    }

    void ProvidedVariableAssociations::AssociateVariableFromInputData(WmTestData* data)
    {
        AssociateVariable("distance",    (*data)["distance"]);
        AssociateVariable("metric_data", (*data)["metric_data"]);
        AssociateVariable("x",           (*data)["x-coordinate"]);
        AssociateVariable("coord_sys",   (*data)["coord_sys"]);
        AssociateVariable("p",           (*data)["pressure"]);
        AssociateVariable("u",           (*data)["u-velocity"]);
        AssociateVariable("v",           (*data)["v-velocity"]);
        AssociateVariable("w",           (*data)["w-velocity"]);
        AssociateVariable("T",           (*data)["Temperature"]);
        AssociateVariable("turb",        (*data)["nu_tilde"]);
        AssociateVariable("rho",         (*data)["density"]);
        AssociateVariable("mu_lam",      (*data)["laminar_viscosity"]);
    }

    bool ProvidedVariableAssociations::Associated(const char* name, double** buffer_addr)
    {
        int hash = get_hash_id(name);
        for (int i = 0; i < num_bufs; i++)
        {
            if (hash == name_hashes[i])
            {
                *buffer_addr = buffer_pointers[i];
                return true;
            }
        }
        return false;
    }

    int ProvidedVariableAssociations::get_hash_id(const char* s)
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
