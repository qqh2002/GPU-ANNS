CXX=g++
DISTTYPE=__USE_COS_DIST

all : anns_cpu

anns_cpu : main.cc config.h graph.h parser_dense.h parser.h data.h logger.h
	$(CXX) main.cc -o anns_cpu -std=c++11 -O3 -march=native
