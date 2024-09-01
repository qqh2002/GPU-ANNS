CXX=g++
NVCC=nvcc
DISTTYPE=USE_L2_DIST_
FLAG_DEBUG=-O3

all : query

query : main.cu 
	$(NVCC) \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_75,code=compute_75 \
	-std=c++11 main.cu $(FLAG_DEBUG) -ccbin=$(CXX) -o query -Xptxas -v \
	-D$(DISTTYPE)