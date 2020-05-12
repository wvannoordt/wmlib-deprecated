#include <iostream>
#include "wall_model_worker.h"
#include "mpi.h"
#include <cmath>

int main(void)
{
	bool ignorefailure = false;
	MPI_Init(NULL, NULL);
	std::cout << "\n\n";
	std::cout << "Testing source in file " << __FILE__ << ":" << std::endl;
	wall_model_module::wmm_init(MPI_COMM_WORLD, false);

	double error = wall_model_module::wmm_run_test_case("bump-ode.dat");

	bool failed = std::abs(error) > 1 || (error != error);
	std::cout << " >>> [Bump, ODE]               Total error: " << std::abs(error) << (!failed ? " (success)" : (ignorefailure?" (failure, ignored)":" (failure)")) << std::endl;
	wall_model_module::wmm_finalize();
	MPI_Finalize();
	std::cout << "\n\n";
	if (ignorefailure) return 0;
	return failed?1:0;
}
