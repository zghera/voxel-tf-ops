#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "../../utils/utils.cuh"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

/*Function: Get how many points in each voxel grid.
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
  Note: Slight adaptation from original implementation.
*/
__global__ void GridStatsKernel(int b, int n, int r, int r2, int r3,
                                const int* coords, int* ind, int* cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    ind[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    if (ind[i] < r3) {
      atomicAdd(cnt + ind[i], 1);
    }
  }
  __syncthreads();
}

/*Function: Average pool voxelization (forward).
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
  Note: Same as the original implementation.
*/
__global__ void AvgVoxForwardKernel(int b, int c, int n, int s,
                                    const int* ind, const int* cnt,
                                    const float* feat, float* out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(out + j * s + pos, feat[j * n + i] * div_cur_cnt);
      }
    }
  }
  __syncthreads();
}

/*Function: Average pool voxelization (backward).
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
  Note: Same as the original implementation.
*/
__global__ void AvgVoxBackwardKernel(int b, int c, int n, int r3,
                                    const int* ind, const int* cnt,
                                    const float* grad_y, float* grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void AvgVoxForwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r, int r2, int r3,
    const int* coords, const float* features, int* ind, int* cnt, float* out) {
  cudaMemset(ind, 0, b*n*sizeof(ind));
  cudaMemset(cnt, 0, b*r3*sizeof(cnt));
  cudaMemset(out, 0, b*c*r3*sizeof(out));

  TF_CHECK_OK(GpuLaunchKernel(GridStatsKernel,
      b, optimal_num_threads(n), 0, d.stream(),
      b, n, r, r2, r3, coords, ind, cnt));
  TF_CHECK_OK(GpuLaunchKernel(AvgVoxForwardKernel,
    b, optimal_num_threads(n), 0, d.stream(),
    b, c, n, r3, ind, cnt, features, out));
}

void AvgVoxBackwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r3,
    const int* ind, const int* cnt, const float* grad_dy, float* grad_dx) {
  cudaMemset(grad_dx, 0, b*c*n*sizeof(grad_dx));

  TF_CHECK_OK(GpuLaunchKernel(AvgVoxBackwardKernel,
    b, optimal_num_threads(n), 0, d.stream(),
    b, c, n, r3, ind, cnt, grad_dy, grad_dx));
}

}  // end namespace tensorflow

#endif // GOOGLE_CUDA
