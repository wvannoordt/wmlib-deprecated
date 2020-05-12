#include "wm_lib_typedef.h"

#ifndef LIN_ALG_TYPE_H
#define LIN_ALG_TYPE_H


template <typename REALPOINTER>
struct MAT_tridiag
{
	int dim;
	Buffer<REALPOINTER> diag;
	Buffer<REALPOINTER> sub;
	Buffer<REALPOINTER> sup;
};

template <typename REALPOINTER, const int SYSTEM_EQUATIONS>
struct LINSYS_block_tridiag
{
	int blockdim;
	const int num_eqs = SYSTEM_EQUATIONS;
	MAT_tridiag<REALPOINTER> block_matrices[SYSTEM_EQUATIONS*SYSTEM_EQUATIONS];
	Buffer<REALPOINTER> rhs[SYSTEM_EQUATIONS];
};

template <typename REALNUMBER, const int DIMENSION>
struct SmallMatrix
{
	int idx=0;
	REALNUMBER data[DIMENSION*DIMENSION];
	REALNUMBER& operator () (int i, int j) {return data[i + j*DIMENSION];}
	const REALNUMBER& operator () (int i, int j) const {return data[i + j*DIMENSION];}
	REALNUMBER& operator () (int i) {return data[i];}
	const REALNUMBER& operator () (int i) const {return data[i];}
};

#endif
