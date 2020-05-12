#ifndef WM_VIZ_H
#define WM_VIZ_H
#include "wall_model_worker.h"
#include "wm_lib_typedef.h"
#include "wm_common.h"
#include "lin_alg_types.h"

namespace wall_model_module
{
    template <const int N> void visualize_linear_system(const char* filename, LINSYS_block_tridiag<double*, N>* sys, const int width, const int height);
    void processelem(double current, double min_neg_abs, double max_neg_abs, double min_pos_abs, double max_pos_abs, int i, int j, int* lines_h, int* lines_w, int nbin_height, int nbin_width, double* red, double* grn, double* blu, int width, int height);
}

#endif
