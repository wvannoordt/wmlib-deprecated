
ifeq (${CUDA_ENABLE},1)
LCUDA := -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudadevrt -lcudart
else
LCUDA := 
endif

CC := mpicxx
MAIN := $(wildcard *.cc)
MAINO := obj/main.o
EXE := program

final: main
	@${CC} ${MAINO} ${LCUDA} -o ${EXE} ${WMLIB_INCLUDE_CONFIG}

main: setup
	@${CC} -c ${MAIN} -o ${MAINO} ${WMLIB_INCLUDE_CONFIG}

setup:
	@mkdir -p ./obj

clean:
	@-rm -f ${EXE}
	@-rm -rf ./obj
	
test: clean final
	@./${EXE}
