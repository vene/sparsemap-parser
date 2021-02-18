DYNET_PATH ?= /home/vlad/code/dynet
EIGEN_PATH ?= /home/vlad/code/eigen
AD3_PATH ?= /home/vlad/code/ad3

OBJS = $(AD3_PATH)/examples/cpp/parsing/FactorTree.o 
CC = g++
DEBUG = -g
INCLUDES = -I$(DYNET_PATH) -I$(EIGEN_PATH) -I$(AD3_PATH) -I$(AD3_PATH)/examples/cpp
LIBS = -L$(AD3_PATH)/ad3 
CFLAGS = -O3 -Wall -Wno-sign-compare -c -fmessage-length=0 $(INCLUDES) -DEIGEN_FAST_MATH -fPIC -fno-finite-math-only -Wno-missing-braces -std=c++11 -funroll-loops
LFLAGS = $(LIBS) -ldynet -lad3

ifdef MKL_PATH
INCLUDES += -I$(MKL_PATH)/include
LIBS += -L$(MKL_PATH)/lib/intel64
LFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
CFLAGS += -DEIGEN_USE_MKL_ALL
endif

LIBS_GPU = $(LIBS) -L$(DYNET_PATH)/build-cuda/dynet/ 
LIBS_CPU = $(LIBS) -L$(DYNET_PATH)/build/dynet/ 
LFLAGS_GPU = $(LIBS_GPU) $(LFLAGS)
LFLAGS_CPU = $(LIBS_CPU) $(LFLAGS) 

all: parser-cpu parser-gpu

parser-cpu: $(OBJS) mst.o mst_losses-cpu.o parser-cpu.o 
	$(CC) $(OBJS) mst.o mst_losses-cpu.o parser-cpu.o $(LFLAGS_CPU) -o parser-cpu

parser-cpu.o : parser.cc utils.h args.h parser.h mst_losses.h scorers.h
	$(CC) $(CFLAGS) parser.cc -o parser-cpu.o 

parser-gpu: $(OBJS) parser-gpu.o 
	$(CC) $(OBJS) parser-gpu.o $(LFLAGS_GPU) -o parser-gpu

parser-gpu.o : parser.cc utils.h args.h parser.h mst_losses.h scorers.h
	$(CC) $(CFLAGS) -DUSING_CUDA parser.cc -o parser-gpu.o 

mst_losses-cpu.o : mst_losses.cc mst_losses.h mst.h
	$(CC) $(CFLAGS) mst_losses.cc -o mst_losses-cpu.o

mst.o: mst.cc mst.h
	$(CC) $(CFLAGS) mst.cc -o mst.o

test_mst : mst.o mst_losses-cpu.o test_mst.o
	$(CC) $(OBJS) mst.o test_mst.o mst_losses-cpu.o $(LFLAGS_CPU) -o test_mst

test_mst.o : test_mst.cc mst.h mst_losses.h
	$(CC) $(CFLAGS) test_mst.cc -o test_mst.o 

clean:
	rm -f *.o *~ parser-cpu parser-gpu 
