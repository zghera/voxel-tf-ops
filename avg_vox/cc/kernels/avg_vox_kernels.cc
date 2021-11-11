#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "avg_vox.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// OpKernel definition.
template <typename Device>
class AvgVoxForwardOp : public OpKernel {
 public:
  explicit AvgVoxForwardOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // // Grab the input tensor
    // const Tensor& input_tensor = context->input(0);

    // // Create an output tensor
    // Tensor* output_tensor = NULL;

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

    // OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
    //                                                  &output_tensor));

    // Do the computation.
    AvgVoxForwardFunctor<Device>()(
        context->eigen_device<Device>()
        // static_cast<int>(input_tensor.NumElements()),
        // input_tensor.flat<T>().data(),
        // output_tensor->flat<T>().data()
    );
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
extern template struct AvgVoxForwardFunctor<GPUDevice>;           
REGISTER_KERNEL_BUILDER(                                      
    Name("AvgVoxForward").Device(DEVICE_GPU), AvgVoxForwardOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // end namespace functor
}  // end namespace tensorflow
