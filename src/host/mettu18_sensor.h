#ifndef WM_METTU_SENS
#define WM_METTU_SENS
#include "wall_model_worker.h"

namespace wall_model_module
{
    void mettu18_initialize(void);
    void mettu18_compute_sensor_values(void);
    void mettu18_setvar(const char* varname, double** buf);
    void copy_avg(double* a, double* b);
    void copy_avg_sq(double* a, double* b);
    void zero_init(double* a);
    void mettu18_compute_average(double* phibar, double* phi);
    void mettu18_compute_average_sqaure(double* phibar, double* phi);
}


#endif
