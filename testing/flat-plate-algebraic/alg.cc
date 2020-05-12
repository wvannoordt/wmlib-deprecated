#include <iostream>
#include "wall_model_worker.h"
#include "mpi.h"
#include <cmath>

int main(void)
{
	MPI_Init(NULL, NULL);
	std::cout << "\n\n";
	std::cout << "Testing source in file " << __FILE__ << ":" << std::endl;
	wall_model_module::wmm_init(MPI_COMM_WORLD, false);

	double error = wall_model_module::wmm_run_test_case("flatplate-alg.dat");
	
	bool failed = std::abs(error) > 1e-3 || (error != error);
	std::cout << " >>> [Flat-Plate, Analytical]  Total error: " << std::abs(error) << (!failed ? " (success)" : " (failure)") << std::endl;
	wall_model_module::wmm_finalize();
	MPI_Finalize();
	std::cout << "\n\n";
	return failed?1:0;
}
