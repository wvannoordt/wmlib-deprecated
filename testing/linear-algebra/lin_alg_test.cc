#include <iostream>
#include "wall_model_worker.h"
#include <cmath>
#include <stdlib.h>

void outputvalidmat(MAT_tridiag<double*>* mat)
{
	std::cout << "Diag:\n";
	for (int i = 0; i < mat->dim; i++)
	{
		std::cout << mat->diag.base[i] << std::endl;
	}
	std::cout << "sup:\n";
	for (int i = 0; i < mat->dim-1; i++)
	{
		std::cout << mat->sup.base[i] << std::endl;
	}
	std::cout << "sub:\n";
	for (int i = 0; i < mat->dim-1; i++)
	{
		std::cout << mat->sub.base[i] << std::endl;
	}
}
void outputvalidbuf(Buffer<double*>* buf, int n)
{
	std::cout << "buf:\n";
	for (int i = 0; i < n; i++)
	{
		std::cout << buf->base[i] << std::endl;
	}
}

template <const int DIM>
void outputvalidsys(LINSYS_block_tridiag<__buffertype*, DIM>* linsys)
{
	int n = linsys->block_matrices[0].dim;
	for (int i = 0; i < DIM; i++)
	{
		std::cout << "rhs " << i << std::endl;
		outputvalidbuf(&(linsys->rhs[i]), n);
	}
	for (int i = 0; i < DIM*DIM; i++)
	{
		std::cout << "lhs " << i << std::endl;
		outputvalidmat(&(linsys->block_matrices[i]));
	}
}

void makebuf(Buffer<double*>* buf, const int& N)
{
	buf->size = N*sizeof(double);
	buf->base = (double*)malloc(buf->size);
	buf->stride = 1;
}

void killbuf(Buffer<double*>* buf)
{
	free(buf->base);
}

template <const int DIM>
void makesys(LINSYS_block_tridiag<__buffertype*, DIM>* linsys, int N)
{
	for (int i = 0; i < DIM; i++)
	{
		makebuf(&(linsys->rhs[i]), N);
	}
	for (int i = 0; i < DIM*DIM; i++)
	{
		makebuf(&(linsys->block_matrices[i].diag), N);
		makebuf(&(linsys->block_matrices[i].sub), N-1);
		makebuf(&(linsys->block_matrices[i].sup), N-1);
		linsys->block_matrices[i].dim = N;
	}
}

template <const int DIM>
void killsys(LINSYS_block_tridiag<double*, DIM>* linsys)
{
	for (int i = 0; i < DIM; i++)
	{
		killbuf(&(linsys->rhs[i]));
	}
	for (int i = 0; i < DIM*DIM; i++)
	{
		killbuf(&(linsys->block_matrices[i].diag));
		killbuf(&(linsys->block_matrices[i].sub));
		killbuf(&(linsys->block_matrices[i].sup));
	}
}

double test_tridiag(void)
{
	#include "thomas.hpp"
	return error;
}

double test_uppertridiag(void)
{
	//generated in matlab script
	#include "tdma_forward.hpp"
	return error;
}

int main(void)
{
	//test tridiagonal algorithm
	std::cout << "\n\n";
	bool failed = false;
	double error;
	std::cout << "Testing source in file " << __FILE__ << ":" << std::endl;
	error = test_tridiag();
	failed = failed || std::abs(error) > 1e-7 || (error != error);
	std::cout << " >>> [Tridiagonal]             Total error: " << std::abs(error) << ((std::abs(error) < 1e-7 && error == error) ? " (success)" : " (failure)") << std::endl;
	error = test_uppertridiag();
	failed = failed || std::abs(error) > 1e-7 || (error != error);
	std::cout << " >>> [Block Upper Tridiagonal] Total error: " << std::abs(error) << ((std::abs(error) < 1e-7 && error == error) ? " (success)" : " (failure)") << std::endl;
	std::cout << "\n\n";
	return failed?1:0;
}
