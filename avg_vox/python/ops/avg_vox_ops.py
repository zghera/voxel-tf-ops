"""avg_vox ops in python."""

from pathlib import Path
from typing import List, Optional
import tensorflow as tf

from tensorflow.python.framework import ops

__all__ = ["avg_voxelize_forward", "avg_voxelize_backward"]


library_file_path = str((Path(__file__).parent / "_avg_vox_ops.so").resolve())
avg_vox_ops = tf.load_op_library(library_file_path)
avg_voxelize_forward = avg_vox_ops.avg_vox_forward
avg_voxelize_backward = avg_vox_ops.avg_vox_backward


@ops.RegisterGradient("AvgVoxForward")
def _avg_vox_grad(
  op: tf.Operation, grad: List[tf.Tensor]
) -> List[Optional[tf.Tensor]]:
  """The gradients for the `avg_vox` op.

  Args:
    op: The `avg_vox` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `avg_vox` op.

  Returns:
    Gradients with respect to the input of `avg_vox`.
  """
  _, ind, cnt = op.outputs
  out_grad = grad[0]
  input_grad = avg_voxelize_backward(out_grad, ind, cnt)
  return [input_grad, None]
