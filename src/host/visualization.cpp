#include "wall_model_worker.h"
#include "wm_lib_typedef.h"
#include "wm_common.h"
#include "lin_alg_types.h"
#include "visualization.h"
#include "hybrid_computing.h"
#include "debug_output.h"
#include "pngwriter.h"
#include <cmath>
#include <stdio.h>
using std::abs;
using std::log10;

namespace wall_model_module
{
    template void visualize_linear_system<3>(const char*, LINSYS_block_tridiag<double*, 3>*, const int, const int);
    template void visualize_linear_system<2>(const char*, LINSYS_block_tridiag<double*, 2>*, const int, const int);
    template <const int N> void visualize_linear_system(const char* filename, LINSYS_block_tridiag<double*, N>* sys, const int width, const int height)
    {
        double* red = (double*)malloc(width*height*sizeof(double));
        double* grn = (double*)malloc(width*height*sizeof(double));
        double* blu = (double*)malloc(width*height*sizeof(double));
        int nbin_height = N*sys->blockdim;
        int nbin_width = N*sys->blockdim + 1;
        int* lines_h = new int[nbin_height+1];
        int* lines_w = new int[nbin_width+1];

        for (int i = 0; i < nbin_height; i++)
        {
            lines_h[i] = (int)((double)i * height / (double)nbin_height);
        }
        for (int i = 0; i < nbin_width; i++)
        {
            lines_w[i] = (int)((double)i * width / (double)nbin_width);
        }
        lines_h[nbin_height] = height;
        lines_w[nbin_width] = width;
        double min_neg_abs = 1.0e100;
        double max_neg_abs = -1.0e100;
        double min_pos_abs = 1.0e100;
        double max_pos_abs = -1.0e100;
        double* current;
        for (int k = 0; k < N*N; k++)
        {
            current = sys->block_matrices[k].diag.base;
            for (int i = 0; i < sys->blockdim; i++)
            {
                min_neg_abs = (log10(abs(current[i])) < min_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):min_neg_abs;
                max_neg_abs = (log10(abs(current[i])) > max_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):max_neg_abs;
                min_pos_abs = (log10(abs(current[i])) < min_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):min_pos_abs;
                max_pos_abs = (log10(abs(current[i])) > max_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):max_pos_abs;
            }
            current = sys->block_matrices[k].sub.base;
            for (int i = 0; i < sys->blockdim-1; i++)
            {
                min_neg_abs = (log10(abs(current[i])) < min_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):min_neg_abs;
                max_neg_abs = (log10(abs(current[i])) > max_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):max_neg_abs;
                min_pos_abs = (log10(abs(current[i])) < min_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):min_pos_abs;
                max_pos_abs = (log10(abs(current[i])) > max_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):max_pos_abs;
            }
            current = sys->block_matrices[k].sup.base;
            for (int i = 0; i < sys->blockdim-1; i++)
            {
                min_neg_abs = (log10(abs(current[i])) < min_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):min_neg_abs;
                max_neg_abs = (log10(abs(current[i])) > max_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):max_neg_abs;
                min_pos_abs = (log10(abs(current[i])) < min_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):min_pos_abs;
                max_pos_abs = (log10(abs(current[i])) > max_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):max_pos_abs;
            }
        }
        for (int k = 0; k < N; k++)
        {
            current = sys->rhs[k].base;
            for (int i = 0; i < sys->blockdim; i++)
            {
                min_neg_abs = (log10(abs(current[i])) < min_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):min_neg_abs;
                max_neg_abs = (log10(abs(current[i])) > max_neg_abs && current[i] < 0 && current[i] == current[i])?log10(abs(current[i])):max_neg_abs;
                min_pos_abs = (log10(abs(current[i])) < min_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):min_pos_abs;
                max_pos_abs = (log10(abs(current[i])) > max_pos_abs && current[i] > 0 && current[i] == current[i])?log10(abs(current[i])):max_pos_abs;
            }
        }
        std::fill_n(red, width*height, 0.3);
        std::fill_n(grn, width*height, 0.3);
        std::fill_n(blu, width*height, 0.3);


        for (int k = 0; k < N*N; k++)
        {
            int delta_i = sys->blockdim * (k / N);
            int delta_j = sys->blockdim * (k % N);
            current = sys->block_matrices[k].diag.base;
            for (int i = 0; i < sys->blockdim; i++)
            {
                processelem(current[i], min_neg_abs, max_neg_abs, min_pos_abs, max_pos_abs, i+delta_i, i+delta_j, lines_h, lines_w, nbin_height, nbin_width, red, grn, blu, width, height);
            }
            current = sys->block_matrices[k].sub.base;
            for (int i = 0; i < sys->blockdim-1; i++)
            {
                processelem(current[i], min_neg_abs, max_neg_abs, min_pos_abs, max_pos_abs, i+1+delta_i, i+delta_j, lines_h, lines_w, nbin_height, nbin_width, red, grn, blu, width, height);
            }
            current = sys->block_matrices[k].sup.base;
            for (int i = 0; i < sys->blockdim-1; i++)
            {
                processelem(current[i], min_neg_abs, max_neg_abs, min_pos_abs, max_pos_abs, i+delta_i, i+1+delta_j, lines_h, lines_w, nbin_height, nbin_width, red, grn, blu, width, height);
            }
        }
        for (int k = 0; k < N; k++)
        {
            current = sys->rhs[k].base;
            for (int i = 0; i < sys->blockdim; i++)
            {
                processelem(current[i], min_neg_abs, max_neg_abs, min_pos_abs, max_pos_abs, i + k*sys->blockdim, N*sys->blockdim, lines_h, lines_w, nbin_height, nbin_width, red, grn, blu, width, height);
            }
        }

        pngwriter writer;
        writer.write_png(filename, width, height, red, grn, blu);

        delete [] lines_h;
        delete [] lines_w;
        free(red);
        free(grn);
        free(blu);
    }

    void processelem(double current, double min_neg_abs, double max_neg_abs, double min_pos_abs, double max_pos_abs, int i, int j, int* lines_h, int* lines_w, int nbin_height, int nbin_width, double* red, double* grn, double* blu, int width, int height)
    {
        double r, g, b;
        double r0, g0, b0, r1, g1, b1;
        double minval, maxval, fvalue;
        bool isneg = current < 0;
        bool iszero = abs(current)<1e-90;
        if (isneg)
        {
            r0 = 0.0;
            g0 = 0.0;
            b0 = 0.0;
            r1 = 0.0;
            g1 = 0.0;
            b1 = 1.0;
            minval = min_neg_abs;
            maxval = max_neg_abs;
        }
        else
        {
            r0 = 0.0;
            g0 = 0.0;
            b0 = 0.0;
            r1 = 1.0;
            g1 = 0.0;
            b1 = 0.0;
            minval = min_pos_abs;
            maxval = max_pos_abs;
        }
        fvalue = log10(abs(current));
        int rowmin = lines_h[i];
        int rowmax = lines_h[i+1];
        int colmin = lines_w[j];
        int colmax = lines_w[j+1];
        double theta = (fvalue - minval)/(maxval-minval);
        bool isnan = current != current;
        for (int row = rowmin; row < rowmax; row++)
        {
            for (int col = colmin; col < colmax; col++)
            {
                if (isnan)
                {
                    *(red + row*width + col) = 1.0;
                    *(grn + row*width + col) = 1.0;
                    *(blu + row*width + col) = 0.0;
                }
                else if (iszero)
                {
                    *(red + row*width + col) = 1.0;
                    *(grn + row*width + col) = 1.0;
                    *(blu + row*width + col) = 1.0;
                }
                else
                {
                    *(red + row*width + col) = r1*theta + r0*(1.0 - theta);
                    *(grn + row*width + col) = g1*theta + g0*(1.0 - theta);
                    *(blu + row*width + col) = b1*theta + b0*(1.0 - theta);
                }
            }
        }

    }
}
