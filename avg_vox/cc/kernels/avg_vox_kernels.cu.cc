#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "avg_vox.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__global__ void AvgVoxForwardCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    // out[i] = 2 * ldg(in + i);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
//
// See core/util/cuda_kernel_helper.h for example of computing
// block count and thread_per_block count.
struct AvgVoxForwardFunctor<GPUDevice> {
  void operator()(const GPUDevice& d) {
    int block_count = 1024;
    int thread_per_block = 20;
    AvgVoxForwardCudaKernel<<<block_count, thread_per_block, 0, d.stream()>>>(
      // size, in, out
    );
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
