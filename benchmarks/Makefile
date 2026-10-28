cutensor: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cutensor_benchmark \
	   -I${TAPP_INCLUDE_PATH} \
	   -I${TAPP_API_INCLUDE_PATH} \
	   -L${TAPP_CUTENSOR_LIBRARY_PATH} \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cutensor -Xcompiler -fopenmp
cublas_eager: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_eager_benchmark \
	   -I${TAPP_INCLUDE_PATH} \
	   -L${TAPP_CUBLAS_LIBRARY_PATH} \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
cublas: benchmark.cu                                                                                                               
	nvcc benchmark.cu -o cublas_benchmark -DUSE_CUBLAS=1\
	   -I${TAPP_API_INCLUDE_PATH} \
	   -I${TAPP_CUBLAS_INCLUDE_PATH} \
	   -L${TAPP_CUBLAS_LIBRARY_PATH} \
           -arch=sm_100 \
	   -lcurand -lcublas -ltapp-cublas -Xcompiler -fopenmp
all: cublas cublas_eager cutensor
