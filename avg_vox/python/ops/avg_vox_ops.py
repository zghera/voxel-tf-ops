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

from pathlib import Path
from typing import List
import tensorflow as tf

from tensorflow.python.framework import ops

__all__ = ["avg_voxelize_forward"]


library_file_path = str((Path(__file__).parent / "_avg_vox_ops.so").resolve())
avg_vox_ops = tf.load_op_library(library_file_path)
avg_voxelize_forward = avg_vox_ops.avg_vox_forward
avg_voxelize_backward = avg_vox_ops.avg_vox_backward


@ops.RegisterGradient("AvgVoxForward")
def _avg_vox_grad(op: tf.Operation, grad: List[tf.Tensor]) -> List[tf.Tensor]:
  """The gradients for the `avg_vox` op.

  Args:
    op: The `avg_vox` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `avg_vox` op.

  Returns:
    Gradients with respect to the input of `avg_vox` for the `out` output.
  """
  _, ind, cnt = op.outputs
  out_grad = grad[0]
  features_grad = avg_voxelize_backward(out_grad, ind, cnt)
  return [features_grad, None, None]
