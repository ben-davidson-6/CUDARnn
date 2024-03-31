CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++-12
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=rnn
INCLUDES = -I${CUDA_PATH}/samples/common/inc
# Openmp
LIBRARIES += -lcublas
ALL_CCFLAGS += -std=c++11 $(INCLUDES) -L${CUDA_PATH}/lib64

all : clean ${TARGET}
rnn_host.o: rnn.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<
rnn_kernel.o: rnn.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS)  -o $@ -c $<
rnn: rnn_kernel.o rnn_host.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS)  -o $@ $+ $(LIBRARIES)
clean:
	rm -f ${TARGET} *.o
