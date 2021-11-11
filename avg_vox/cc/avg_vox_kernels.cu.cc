#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__ void AvgVoxForwardKernel() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 0;
       i += blockDim.x * gridDim.x) {

  }
}

// Define the GPU implementation that launches the CUDA kernel.
//
// See core/util/gpu_kernel_helper.h for example of computing
// block count and thread_per_block count.
Status AvgVoxForwardKernelLauncher(const GPUDevice& d) {
  int block_count = 1024;
  int thread_per_block = 20;
  return GpuLaunchKernel(AvgVoxForwardKernel,
    block_count, thread_per_block, 0, d.stream()
  );
}

}  // end namespace tensorflow

#endif // GOOGLE_CUDA
