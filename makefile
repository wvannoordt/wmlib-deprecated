LIB_NAME := wmworker

ifdef USE_INTEL
CC_COMMON = nvcc -ccbin=icpc
CC_DEVICE = nvcc -ccbin=icpc
MPICXX_CXX = icpc
else
CC_COMMON = nvcc
CC_DEVICE = nvcc
endif
CC_HOST = mpicxx

ifndef DYNAMIC_LIBRARY
DYNAMIC_LIBRARY := 0
endif

ifndef WM_ALLOW_DEBUG_EXT
WM_ALLOW_DEBUG_EXT := 0
endif

ifndef OPT_LEVEL
OPT_LEVEL := 0
endif

WM_BASEIDIR = $(shell pwd)
WM_SRC_DIR := ${WM_BASEIDIR}/src

WM_LIB_DIR := ${WM_BASEIDIR}/lib
WM_OBJ_DIR := ${WM_BASEIDIR}/obj
WM_HDR_DIR := ${WM_BASEIDIR}/include

ifndef BITCART_SRC
BITCART_SRC := ${BITCART_SOURCE_PATH}
endif

WM_IFLAGS := -I${WM_HDR_DIR} -I${BITCART_SRC}/navier_stokes_newest

export WMLIB_INCLUDE_CONFIG=-I${WM_HDR_DIR} -I${BITCART_SRC}/navier_stokes_newest -L${WM_LIB_DIR} -l${LIB_NAME}

SRC_FILES_COMMON        := $(wildcard ${WM_SRC_DIR}/common/*.cpp)
SRC_FILES_HOST          := $(wildcard ${WM_SRC_DIR}/host/*.cpp)
SRC_FILES_KERNEL        := $(wildcard ${WM_SRC_DIR}/kernel/*.cu)
SRC_FILES_HYBRID_HOST   := $(wildcard ${WM_SRC_DIR}/numerics/*.cpp)
SRC_FILES_HYBRID_DEVICE := $(wildcard ${WM_SRC_DIR}/numerics/*.cpp)

HDR_FILES_COMMON        := $(wildcard ${WM_SRC_DIR}/common/*.h)
HDR_FILES_HOST          := $(wildcard ${WM_SRC_DIR}/host/*.h)
HDR_FILES_KERNEL        := $(wildcard ${WM_SRC_DIR}/kernel/*.h)
HDR_FILES_HYBRID_HOST   := $(wildcard ${WM_SRC_DIR}/numerics/*.h)

HEADER_FILES := ${HDR_FILES_COMMON}
HEADER_FILES += ${HDR_FILES_HOST}
HEADER_FILES += ${HDR_FILES_KERNEL}
HEADER_FILES += ${HDR_FILES_HYBRID_HOST}

OBJ_FILES_COMMON        := $(patsubst ${WM_SRC_DIR}/common/%.cpp,$(WM_OBJ_DIR)/%.o,$(SRC_FILES_COMMON))
OBJ_FILES_HOST          := $(patsubst ${WM_SRC_DIR}/host/%.cpp,$(WM_OBJ_DIR)/%.o,$(SRC_FILES_HOST))
OBJ_FILES_KERNEL        := $(patsubst ${WM_SRC_DIR}/kernel/%.cu,$(WM_OBJ_DIR)/%.o,$(SRC_FILES_KERNEL))
OBJ_FILES_HYBRID_HOST   := $(patsubst ${WM_SRC_DIR}/numerics/%.cpp,$(WM_OBJ_DIR)/%.o,$(SRC_FILES_HYBRID_HOST))
OBJ_FILES_HYBRID_DEVICE := $(patsubst ${WM_SRC_DIR}/numerics/%.cpp,$(WM_OBJ_DIR)/K_%.o,$(SRC_FILES_HYBRID_DEVICE))


TARGET := 
SYSLINK := 
LINK_LOCATION :=

ifeq (${DYNAMIC_LIBRARY}, 1)
TARGET := ${WM_LIB_DIR}/lib${LIB_NAME}.so
SYSLINK := syslink
LINK_LOCATION := /home/wvn/.lib_local
else
TARGET := ${WM_LIB_DIR}/lib${LIB_NAME}.a
SYSLINK :=
LINK_LOCATION := 
endif

ifndef CUDA_ENABLE
CUDA_ENABLE := 0
endif

ifndef BENCHMARKING_ENABLE
BENCHMARKING_ENABLE := 0
endif

ifndef PROBLEM_DIMENSION
PROBLEM_DIMENSION := 2
endif

CU_O_TARGET_NAME := ${WM_OBJ_DIR}/CU.o
ifeq (${CUDA_ENABLE}, 1)
LINK_STEP := link_step
CU_O_TARGET := ${CU_O_TARGET_NAME}
LCUDA := -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudadevrt -lcudart
else
OBJ_FILES_HYBRID_DEVICE := 
CC_COMMON := mpicxx
CUDA_ENABLE := 0
OBJ_FILES_KERNEL := 
CU_O_TARGET := 
LCUDA := 
endif


NVCC_FLAGS := -O${OPTLEVEL} -x cu -rdc=true -Xcompiler -fPIC ${COMPILE_TIME_OPT} -dc
NVCC_DLINK_FLAGS := -Xcompiler -fPIC -rdc=true -dlink
MPICC_FLAGS := -O${OPTLEVEL} -fPIC -fpermissive -std=c++11 -c ${LCUDA}

COMMON_FLAGS := ${NVCC_FLAGS}
ifeq (${CUDA_ENABLE}, 0)
COMMON_FLAGS := ${MPICC_FLAGS}
endif

LZLIB := 
ifeq (${WM_ALLOW_DEBUG_EXT}, 1)
LZLIB := -lz
endif


DO_CLEAN :=
ifeq (1, ${CUDA_ENABLE})
ifeq (,$(wildcard ${CU_O_TARGET_NAME}))
DO_CLEAN := clean
endif
endif

ifeq (0, ${CUDA_ENABLE})
ifneq (,$(wildcard ${CU_O_TARGET_NAME}))
DO_CLEAN := clean
endif
endif

COMPILE_TIME_OPT := -DUSE_CUDA_RUNTIME=1
COMPILE_TIME_OPT += -DCUDA_ENABLE=${CUDA_ENABLE}
COMPILE_TIME_OPT += -DGPU_PRECISION=2
COMPILE_TIME_OPT += -DRESTRICT_C_POINTERS=1
COMPILE_TIME_OPT += -DPROBLEM_DIMENSION=${PROBLEM_DIMENSION}
COMPILE_TIME_OPT += -DBENCHMARKING_ENABLE=${BENCHMARKING_ENABLE}
COMPILE_TIME_OPT += -DWM_DEBUG_OUTPUT=0
COMPILE_TIME_OPT += -DANALYTICAL_JACOBIAN=0
COMPILE_TIME_OPT += -DWM_ALLOW_DEBUG_EXT=${WM_ALLOW_DEBUG_EXT}


.PHONY: final
final: ${DO_CLEAN} setup ${OBJ_FILES_HYBRID_HOST} ${OBJ_FILES_HOST} ${OBJ_FILES_HYBRID_DEVICE} ${OBJ_FILES_COMMON} ${OBJ_FILES_KERNEL} ${LINK_STEP}
	${CC_HOST} -fPIC -shared ${OBJ_FILES_HOST} ${OBJ_FILES_COMMON} ${OBJ_FILES_KERNEL} ${OBJ_FILES_HYBRID_DEVICE} ${OBJ_FILES_HYBRID_HOST} ${CU_O_TARGET} ${WM_IFLAGS} ${COMPILE_TIME_OPT} ${LZLIB} ${LCUDA} -o ${TARGET}
	
$(OBJ_FILES_HYBRID_DEVICE): ${WM_OBJ_DIR}/K_%.o : ${WM_SRC_DIR}/numerics/%.cpp
	${CC_DEVICE} ${NVCC_FLAGS} ${COMPILE_TIME_OPT} ${WM_IFLAGS} $< -o $@
	
$(OBJ_FILES_HYBRID_HOST): ${WM_OBJ_DIR}/%.o : ${WM_SRC_DIR}/numerics/%.cpp
	${CC_HOST} ${MPICC_FLAGS} ${COMPILE_TIME_OPT} ${WM_IFLAGS} $< -o $@

$(OBJ_FILES_HOST): ${WM_OBJ_DIR}/%.o : ${WM_SRC_DIR}/host/%.cpp
	${CC_HOST} ${MPICC_FLAGS} ${COMPILE_TIME_OPT} ${WM_IFLAGS} $< -o $@

$(OBJ_FILES_COMMON): ${WM_OBJ_DIR}/%.o : ${WM_SRC_DIR}/common/%.cpp
	${CC_COMMON} ${COMMON_FLAGS} ${COMPILE_TIME_OPT} ${WM_IFLAGS} $< -o $@
	
${LINK_STEP}:
	${CC_DEVICE} ${NVCC_DLINK_FLAGS} ${COMPILE_TIME_OPT} ${OBJ_FILES_HYBRID_DEVICE} $(OBJ_FILES_COMMON) ${OBJ_FILES_KERNEL} -o ${CU_O_TARGET} -lcudadevrt

$(OBJ_FILES_KERNEL): ${WM_OBJ_DIR}/%.o : ${WM_SRC_DIR}/kernel/%.cu
	${CC_DEVICE} ${COMPILE_TIME_OPT} ${NVCC_FLAGS} ${WM_IFLAGS} $< -o $@
	
syslink:
	ln -sf ${TARGET} ${LINK_LOCATION}

setup:
	-rm -r ${WM_HDR_DIR}
	mkdir -p ${WM_LIB_DIR}
	mkdir -p ${WM_OBJ_DIR}
	mkdir -p ${WM_HDR_DIR}
	@for hdr in ${HEADER_FILES} ; do \
		ln -s $${hdr} -t ${WM_HDR_DIR};\
	done
	
test: final
	@for fldr in testing/* ; do \
                ${MAKE} -C $${fldr} -f makefile -s test || exit 1; \
        done

clean:
	for fldr in testing/* ; do \
	            ${MAKE} -C $${fldr} -f makefile clean ; \
	    done
	-rm -r ${WM_LIB_DIR}
	-rm -r ${WM_OBJ_DIR}
	-rm -r ${WM_HDR_DIR}

