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

REGISTER_OP("TrilinearDevoxForward")
    .Input("features: float")
    .Input("coords: float")
    .Attr("resolution: int")
    .Attr("is_training: bool")
    .Output("outputs: float")
    .Output("indices: int32")
    .Output("weights: float")
    .SetShapeFn([](InferenceContext* c) {
      // Input rank assertions
      ShapeHandle features;
      ShapeHandle coords;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &features));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &coords));

      // Specifying output shapes
      ShapeHandle outputs_shape = c->MakeShape(
          {c->Dim(features,0), c->Dim(features,1), c->Dim(coords,2)}));
      ShapeHandle indices_shape = c->MakeShape(
        {c->Dim(features,0), 8, c->Dim(coords,2)}));
      c->set_output(0, outputs_shape);
      c->set_output(1, indices_shape);
      c->set_output(2, indices_shape);

      return Status::OK();
});

void TrilinearDevoxForwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r, int r2, int r3, bool is_training,
    const float* coords, const float* features,
    int* indices, float* weights, float* outputs);

// OpKernel definition.
class TrilinearDevoxForwardOp : public OpKernel {
 private:
  int resolution_;
  bool is_training_;
 public:
  explicit TrilinearDevoxForwardOp(OpKernelConstruction* context) 
   : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution_));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& features = context->input(0);
    const Tensor& coords = context->input(1);

    // Get shape and resolution integers
    int batches    = features.shape().dim_size(0);
    int channels   = features.shape().dim_size(1);
    int num_points = coords.shape().dim_size(2);
    int r2 = resolution_ * resolution_;
    int r3 = r2 * resolution_;

    // Create output tensors
    Tensor* outputs = NULL;
    Tensor* indices = NULL;
    Tensor* weights = NULL;
    TensorShape outs_shape{batches, channels, num_points};
    TensorShape inds_shape{batches, 8, num_points};
    TensorShape wgts_shape{batches, 8, num_points};
    OP_REQUIRES_OK(context, context->allocate_output(0, outs_shape, &outputs));
    OP_REQUIRES_OK(context, context->allocate_output(1, inds_shape, &indices));
    OP_REQUIRES_OK(context, context->allocate_output(2, wgts_shape, &weights));

    // Do the computation.
    TrilinearDevoxForwardKernelLauncher(context->eigen_device<GPUDevice>(),
        batches, channels, num_points, resolution_, r2, r3, is_training_,
        coords.flat<float>().data(), features.flat<float>().data(),
        indices->flat<float>().data(), weights->flat<int32>().data(),
        outputs->flat<float>().data() );
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TrilinearDevoxForward").Device(DEVICE_GPU),
                        TrilinearDevoxForwardOp);
#endif  // GOOGLE_CUDA

// -----------------------------------------------------------------------

REGISTER_OP("TrilinearDevoxBackward")
    .Input("grad_dy: float")
    .Input("indices: int32")
    .Input("weights: float")
    .Attr("resolution: int")
    .Output("grad_dx: float")
    .SetShapeFn([](InferenceContext* c) {
      // Input rank assertions
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input));

      // Get (resolution ** 3) to set output shapes
      int resolution;
      TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
      int r3 = resolution * resolution * resolution;

      // Specifying output shapes
      ShapeHandle grad_dx_shape = c->MakeShape({
        c->Dim(c->input(0),0), c->Dim(c->input(0),1), r3});
      c->set_output(0, grad_dx_shape);

      return Status::OK();
});

void TrilinearDevoxBackwardKernelLauncher(const GPUDevice& d,
    int b, int c, int n, int r3, const int* indices, const float* weights,
    const float* grad_dy, float* grad_dx) {

// OpKernel definition.
class TrilinearDevoxBackwardOp : public OpKernel {
 private:
  int resolution_;
 public:
  explicit TrilinearDevoxBackwardOp(OpKernelConstruction* context)
   : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("resolution", &resolution_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& grad_dy = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& weights = context->input(2);

    // Get shape and resolution integers
    int batches    = grad_dy.shape().dim_size(0);
    int channels   = grad_dy.shape().dim_size(1);
    int num_points = grad_dy.shape().dim_size(2);
    int r3 = resolution_ * resolution_ * resolution_;

    // Create output tensors
    Tensor* grad_dx = NULL;
    TensorShape grad_dx_shape{batches, channels, r3};
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_dx_shape,
                                                     &grad_dx));

    // Do the computation.
    TrilinearDevoxBackwardKernelLauncher(context->eigen_device<GPUDevice>(),
        batches, channels, num_points, r3,
        indices.flat<int32>().data(), weights.flat<int32>().data(),
        grad_dy.flat<float>().data(), grad_dx->flat<float>().data() );
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TrilinearDevoxBackward").Device(DEVICE_GPU),
                        TrilinearDevoxBackwardOp);
#endif  // GOOGLE_CUDA


}  // end namespace tensorflow
