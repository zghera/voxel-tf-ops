#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("AvgVoxForward")
    .Input("features: float")
    .Input("coords: int32")
    .Attr("resolution: int")
    .Output("out: float")
    .Output("ind: int32")
    .Output("cnt: int32")
    // TODO: Determine why this shape function was throwing following error:
    // ValueError: Shape must be rank 0 but is rank 1250256272 for 
    //   '{{node pvcnn/point_features_branch/pv_conv/voxelization/AvgVoxForward}} = 
    //   AvgVoxForward[resolution=32](sample, pvcnn/point_features_branch/pv_conv/voxelization/Cast)'
    //   with input shapes: [64,9,4096], [64,3,4096].
    // ------------------------------------------------------------------------
    // .SetShapeFn([](InferenceContext* c) {
    //   // Input rank assertions
    //   ShapeHandle features;
    //   ShapeHandle coords;
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &features));
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &coords));

    //   // Get (resolution ** 3) to set some of the output shapes
    //   int resolution;
    //   TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
    //   int r3 = resolution * resolution * resolution;

    //   // Specifying output shapes
    //   ShapeHandle out_shape = c->MakeShape(
    //       {c->Dim(features,0), c->Dim(features,1), r3});
    //   c->set_output(0, out_shape);
    //   c->set_output(1, c->Matrix(c->Dim(features,0), c->Dim(features,2)));
    //   c->set_output(2, c->Matrix(c->Dim(features,0), r3));

    //   return Status::OK();
    // })
    // ------------------------------------------------------------------------
    .Doc(R"doc(
      Average voxelization operation forward pass.
    )doc");

void AvgVoxForwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r, int r2, int r3,
    const int* coords, const float* features, int* ind, int* cnt, float* out);

// OpKernel definition.
class AvgVoxForwardOp : public OpKernel {
 private:
  int resolution_;
 public:
  explicit AvgVoxForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& features = context->input(0);
    const Tensor& coords = context->input(1);

    // Get shape and resolution integers
    int batches    = features.shape().dim_size(0);
    int channels   = features.shape().dim_size(1);;
    int num_points = features.shape().dim_size(2);;
    int r2 = resolution_ * resolution_;
    int r3 = r2 * resolution_;

    // Create output tensors
    Tensor* out = NULL;
    Tensor* ind = NULL;
    Tensor* cnt = NULL;
    TensorShape out_shape{batches, channels, r3};
    TensorShape ind_shape{batches, num_points};
    TensorShape cnt_shape{batches, r3};
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    OP_REQUIRES_OK(context, context->allocate_output(1, ind_shape, &ind));
    OP_REQUIRES_OK(context, context->allocate_output(2, cnt_shape, &cnt));

    // Do the computation.
    AvgVoxForwardKernelLauncher(context->eigen_device<GPUDevice>(),
        batches, channels, num_points, resolution_, r2, r3,
        coords.flat<int32>().data(), features.flat<float>().data(),
        ind->flat<int32>().data(), cnt->flat<int32>().data(),
        out->flat<float>().data() );
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AvgVoxForward").Device(DEVICE_GPU),
                        AvgVoxForwardOp);
#endif  // GOOGLE_CUDA

// -----------------------------------------------------------------------

REGISTER_OP("AvgVoxBackward")
    .Input("grad_dy: float")
    .Input("ind: int32")
    .Input("cnt: int32")
    .Output("grad_dx: float")
    // TODO: Resolve AvgVoxForward SetShapeFn issue before using SetShapeFn.
    // ------------------------------------------------------------------------
    // .SetShapeFn([](InferenceContext* c) {
    //   // Input rank assertions
    //   ShapeHandle input;
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
    //   TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &input));

    //   // Specifying output shapes
    //   ShapeHandle grad_dx_shape = c->MakeShape({
    //     c->Dim(c->input(0),0), c->Dim(c->input(0),1), c->Dim(c->input(1),1)});
    //   c->set_output(0, grad_dx_shape);

    //   return Status::OK();
    // })
    // ------------------------------------------------------------------------
    .Doc(R"doc(
      Average voxelization operation backward pass.
    )doc");

void AvgVoxBackwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r3,
    const int* ind, const int* cnt, const float* grad_dy, float* grad_dx);

// OpKernel definition.
class AvgVoxBackwardOp : public OpKernel {
 public:
  explicit AvgVoxBackwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& grad_dy = context->input(0);
    const Tensor& ind = context->input(1);
    const Tensor& cnt = context->input(2);

    // Get shape and resolution integers
    int batches    = grad_dy.shape().dim_size(0);
    int channels   = grad_dy.shape().dim_size(1);
    int r3         = grad_dy.shape().dim_size(2);
    int num_points = ind.shape().dim_size(1);

    // Create output tensors
    Tensor* grad_dx = NULL;
    TensorShape grad_dx_shape{batches, channels, num_points};
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_dx_shape,
                                                     &grad_dx));

    // Do the computation.
    AvgVoxBackwardKernelLauncher(context->eigen_device<GPUDevice>(),
        batches, channels, num_points, r3,
        ind.flat<int32>().data(), cnt.flat<int32>().data(),
        grad_dy.flat<float>().data(), grad_dx->flat<float>().data() );
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AvgVoxBackward").Device(DEVICE_GPU),
                        AvgVoxBackwardOp);
#endif  // GOOGLE_CUDA


}  // end namespace tensorflow
