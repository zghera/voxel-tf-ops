CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

VOX_CPP_SRCS = avg_vox/cc/avg_vox_kernels.cc 
VOX_CU_SRCS = avg_vox/cc/avg_vox_kernels.cu.cc 
DEVOX_CPP_SRCS = trilinear_devox/cc/trilinear_devox_kernels.cc
DEVOX_CU_SRCS = trilinear_devox/cc/trilinear_devox_kernels.cu.cc

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}


# avg_vox op for GPU
VOX_GPU_ONLY_TARGET_LIB = avg_vox/python/ops/_avg_vox_ops.cu.o
VOX_TARGET_LIB = avg_vox/python/ops/_avg_vox_ops.so

vox_gpu_only: $(VOX_GPU_ONLY_TARGET_LIB)

$(VOX_GPU_ONLY_TARGET_LIB): $(VOX_CU_SRCS)
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -Iutils -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

vox_op: $(VOX_TARGET_LIB)
$(VOX_TARGET_LIB): $(VOX_CPP_SRCS) $(VOX_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

vox_test: avg_vox/python/ops/avg_vox_ops_test.py avg_vox/python/ops/avg_vox_ops.py $(VOX_TARGET_LIB)
	$(PYTHON_BIN_PATH) avg_vox/python/ops/avg_vox_ops_test.py


# trilinear_devox op for GPU
DEVOX_GPU_ONLY_TARGET_LIB = trilinear_devox/python/ops/_trilinear_devox_ops.cu.o
DEVOX_TARGET_LIB = trilinear_devox/python/ops/_trilinear_devox_ops.so

devox_gpu_only: $(DEVOX_GPU_ONLY_TARGET_LIB)

$(DEVOX_GPU_ONLY_TARGET_LIB): $(DEVOX_CU_SRCS)
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -Iutils -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

devox_op: $(DEVOX_TARGET_LIB)
$(DEVOX_TARGET_LIB): $(DEVOX_CPP_SRCS) $(DEVOX_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

devox_test: trilinear_devox/python/ops/trilinear_devox_ops_test.py trilinear_devox/python/ops/trilinear_devox_ops.py $(DEVOX_TARGET_LIB)
	$(PYTHON_BIN_PATH) trilinear_devox/python/ops/trilinear_devox_ops_test.py

clean_vox:
	rm -f $(VOX_GPU_ONLY_TARGET_LIB) $(VOX_TARGET_LIB)

clean_devox:
	rm -f $(DEVOX_GPU_ONLY_TARGET_LIB) $(DEVOX_TARGET_LIB) 

clean:
	rm -f $(VOX_GPU_ONLY_TARGET_LIB) $(VOX_TARGET_LIB) $(DEVOX_GPU_ONLY_TARGET_LIB) $(DEVOX_TARGET_LIB) 
