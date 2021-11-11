#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("AvgVoxForward")
    .Input("features: float")
    .Input("coords: int32")
    .Input("resolution: int32")
    .Output("out: float")
    .Output("ind: int32");
    .Output("cnt: int32");
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Input rank assertions
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &input));

      // Get (resolution ** 3) as a dimension handle
      DimensionHandle s;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &s));
      TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));
      TF_RETURN_IF_ERROR(c->Multiply(s, s, &s));

      // Specifying output shapes
      ShapeHandle outShape = c->MakeShape(
          {c->dim(c->input(0),0), c->Dim(c->input(0),1), s});
      c->set_output(0, outShape);
      c->set_output(1, c->Matrix(c->dim(c->input(0),0),
                                 c->dim(c->input(0),2) ));
      c->set_output(2, c->Matrix(c->dim(c->input(0),0), s));

      return Status::OK();
    });

// REGISTER_OP("AvgVoxBackward")