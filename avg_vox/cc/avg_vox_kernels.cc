#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/framework/tensor_shape.h"
// #include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("AvgVoxForward")
    .Input("features: float")
    // .Input("coords: int32")
    // .Input("resolution: int32")
    .Output("out: float")
    // .Output("ind: int32");
    // .Output("cnt: int32");
    .SetShapeFn([](InferenceContext* c) {
      // Input rank assertions
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &input));
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &input));

      // // Get (resolution ** 3) as a dimension handle
      // DimensionHandle s;
      // TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &s));
      // TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));
      // TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));

      // // Specifying output shapes
      // ShapeHandle outShape = c->MakeShape(
      //     {c->dim(c->input(0),0), c->Dim(c->input(0),1), s});
      // c->set_output(0, outShape);
      // c->set_output(1, c->Matrix(c->dim(c->input(0),0),
      //                            c->dim(c->input(0),2) ));
      // c->set_output(2, c->Matrix(c->dim(c->input(0),0), s));

      return Status::OK();
});

Status AvgVoxForwardKernelLauncher(const GPUDevice& d);

// OpKernel definition.
class AvgVoxForwardOp : public OpKernel {
 public:
  explicit AvgVoxForwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // // Create an output tensor
    Tensor* output_tensor = NULL;

    //   // Get (resolution ** 3) as a dimension handle
    //   DimensionHandle s;
    //   TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &s));
    //   TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));
    //   TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));

    //   // Specifying output shapes
    //   ShapeHandle outShape = c->MakeShape(
    //       {c->dim(c->input(0),0), c->Dim(c->input(0),1), s});
    //   c->set_output(0, outShape);
    //   c->set_output(1, c->Matrix(c->dim(c->input(0),0),
    //                              c->dim(c->input(0),2) ));
    //   c->set_output(2, c->Matrix(c->dim(c->input(0),0), s));

    OP_REQUIRES_OK(context, context->allocate_output(
      0, input_tensor.shape(), &output_tensor));

    // Do the computation.
    OP_REQUIRES_OK(context, AvgVoxForwardKernelLauncher(
        context->eigen_device<GPUDevice>()));
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AvgVoxForward").Device(DEVICE_GPU),AvgVoxForwardOp);
#endif  // GOOGLE_CUDA

// -----------------------------------------------------------------------

// REGISTER_OP("AvgVoxBackward")
// ...


}  // end namespace tensorflow
