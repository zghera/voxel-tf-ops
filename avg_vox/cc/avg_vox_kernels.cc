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
    .SetShapeFn([](InferenceContext* c) {
      // Input rank assertions
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &input));

      // Get (resolution ** 3) to set some of the output shapes
      int resolution;
      TF_RETURN_IF_ERROR(c->GetAttr("resolution", &resolution));
      int r3 = resolution * resolution * resolution;

      // Specifying output shapes
      ShapeHandle outShape = c->MakeShape(
          {c->Dim(c->input(0),0), c->Dim(c->input(0),1), r3});
      c->set_output(0, outShape);
      c->set_output(1, c->Matrix(c->Dim(c->input(0),0),
                                 c->Dim(c->input(0),2) ));
      c->set_output(2, c->Matrix(c->Dim(c->input(0),0), r3));

      return Status::OK();
});

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
    int batches   = features.shape().dim_size(0);
    int channels  = features.shape().dim_size(1);;
    int numPoints = features.shape().dim_size(2);;
    int r2 = resolution_ * resolution_;
    int r3 = r2 * resolution_;

    // Create output tensors
    Tensor* out = NULL;
    Tensor* ind = NULL;
    Tensor* cnt = NULL;
    TensorShape outShape{batches, channels, r3};
    TensorShape indShape{batches, numPoints};
    TensorShape cntShape{batches, r3};
    OP_REQUIRES_OK(context, context->allocate_output(0, outShape, &out));
    OP_REQUIRES_OK(context, context->allocate_output(1, indShape, &ind));
    OP_REQUIRES_OK(context, context->allocate_output(2, cntShape, &cnt));

    // Do the computation.
    AvgVoxForwardKernelLauncher(context->eigen_device<GPUDevice>(),
        batches, channels, numPoints, resolution_, r2, r3,
        coords.flat<int32>().data(), features.flat<float>().data(),
        ind->flat<int32>().data(), cnt->flat<int32>().data(),
        out->flat<float>().data() );
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
