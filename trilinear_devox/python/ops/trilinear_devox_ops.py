"""trilinear_devox ops in python."""

from pathlib import Path
from typing import List, Optional
import tensorflow as tf

from tensorflow.python.framework import ops

__all__ = ["trilinear_devoxelize_forward", "trilinear_devoxelize_backward"]


library_file_path = str(
  (Path(__file__).parent / "_trilinear_devox_ops.so").resolve()
)
trilinear_devox_ops = tf.load_op_library(library_file_path)
trilinear_devoxelize_forward = trilinear_devox_ops.trilinear_devox_forward
trilinear_devoxelize_backward = trilinear_devox_ops.trilinear_devox_backward


@ops.RegisterGradient("TrilinearDevoxForward")
def _trilinear_devox_grad(
  op: tf.Operation, grad: List[tf.Tensor]
) -> List[Optional[tf.Tensor]]:
  """The gradients for the `trilinear_devox` op.

  Args:
    op: The `trilinear_devox` `Operation` that we are differentiating, which
      we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `trilinear_devox` op.

  Returns:
    Gradients with respect to the input of `trilinear_devox`.
  """
  _, indices, weights = op.outputs
  out_grad = grad[0]
  res = op.get_attr("resolution")
  input_grad = trilinear_devoxelize_backward(out_grad, indices, weights, res)
  return [input_grad, None]
