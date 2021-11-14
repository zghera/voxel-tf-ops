#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "../../utils/utils.cuh"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

/*Function: Trilinear devoxlization (forward).
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r   : voxel resolution
    r2  : r ** 2
    r3  : r ** 3
    coords : the coordinates of points, FloatTensor[b, 3, n]
    feat   : features, FloatTensor[b, c, r3]
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    outs   : outputs, FloatTensor[b, c, n]
  Note: 
*/
__global__ void TrilinearDevoxForwardKernel(
  int b, int c, int n, int r, int r2, int r3, bool is_training,
  const float* coords, const float* feat, int* inds, float* wgts, float* outs) {
}

/*Function: Trilinear devoxlization (backward).
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    r3  : voxel cube size = voxel resolution ** 3
    inds   : the voxel indices of point cube, IntTensor[b, 8, n]
    wgts   : weight for trilinear interpolation, FloatTensor[b, 8, n]
    grad_y : grad outputs, FloatTensor[b, c, n]
    grad_x : grad inputs, FloatTensor[b, c, r3]
  Note: Same as the original implementation.
*/
__global__ void TrilinearDevoxBackwardKernel(int b, int c, int n, int r3,
                                    const int* inds, const float* wgts,
                                    const float* grad_y, float* grad_x) {
}

void TrilinearDevoxForwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r, int r2, int r3, bool is_training,
    const float* coords, const float* features,
    int* indices, float* weights, float* outputs) {
  cudaMemset(indices, 0, b*8*n*sizeof(indices));
  cudaMemset(weights, 0, b*8*sizeof(weights));
  cudaMemset(outputs, 0, b*c*n*sizeof(outputs));

  TF_CHECK_OK(GpuLaunchKernel(TrilinearDevoxForwardKernel,
      b, optimal_num_threads(n), 0, d.stream(),
      b, c, n, r, r2, r3, is_training,
      coords, features, indices, weights, outputs));
}

void TrilinearDevoxBackwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r3, const int* indices, const float* weights,
    const float* grad_dy, float* grad_dx) {
  cudaMemset(grad_dx, 0, b*c*r3*sizeof(grad_dx));

  TF_CHECK_OK(GpuLaunchKernel(TrilinearDevoxBackwardKernel,
    b, optimal_num_threads(n), 0, d.stream(),
    b, c, n, r3, indices, weights, grad_dy, grad_dx));
}

}  // end namespace tensorflow

#endif // GOOGLE_CUDA
