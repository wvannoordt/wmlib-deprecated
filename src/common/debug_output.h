#ifndef DEBOUG_OUT_H
#define DEBOUG_OUT_H
#include "hybrid_computing.h"
#include "wm_lib_typedef.h"

#define ASSERTKILL(mycode) std::cout << mycode << std::endl; std::cout << "Terminate generated in file " << __FILE__ << ", line " << __LINE__ << std::endl; abort();

#if (__cpu)
#define CPUDEBUGOUT(mystuff) std::cout << __FILE__ << ":" << __LINE__<< ": " << mystuff << std::endl
#define CPUWAIT std::cout << "Pause generated in file " << __FILE__ << ", line " << __LINE__ << std::endl; std::cin.get()
#else
#define CPUDEBUGOUT(mystuff)
#define CPUWAIT
#endif

#endif
