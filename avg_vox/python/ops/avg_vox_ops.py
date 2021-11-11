# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""avg_vox ops in python."""

import tensorflow as tf

from tensorflow.python.framework import ops
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import gen_array_ops
# from tensorflow.python.ops import gen_math_ops
# from tensorflow.python.ops import gen_nn_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import sparse_ops
# from tensorflow.python.ops import special_math_ops

__all__ = ["avg_voxelize_forward"]

avg_vox_ops = tf.load_op_library("./_avg_vox_ops.so")
avg_voxelize_forward = avg_vox_ops.avg_vox_forward
# avg_voxelize_backward = avg_vox_ops.avg_vox_backward

@ops.RegisterGradient("AvgVoxForward")
def _avg_vox_grad(op, grad):
  """The gradients for the `avg_vox` op.

  Args:
    op: The `avg_vox` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `avg_vox` op.

  Returns:
    Gradients with respect to the input of `avg_vox`.
  """
#   to_zero = op.inputs[0]
#   shape = array_ops.shape(to_zero)
#   index = array_ops.zeros_like(shape)
#   first_grad = array_ops.reshape(grad, [-1])[0]
#   to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
#   return [to_zero_grad]  # List of one Tensor, since we have one input
  return [None]