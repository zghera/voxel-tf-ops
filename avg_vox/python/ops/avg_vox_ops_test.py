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
"""Tests for avg_vox ops."""
import tensorflow as tf

try:
  from avg_vox.python.ops import avg_voxelize_forward
except ImportError:
  from avg_vox_ops import avg_voxelize_forward


class AvgVoxTest(tf.test.TestCase):
  def test_avg_voxelize_forward(self):
    with tf.device("/device:GPU:0"):
      # [B, C, N] = [2, 5, 4]
      features = tf.constant([
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],
      ], dtype=tf.float32)
      # [B, 3, N] = [2, 3, 4]
      coords = tf.constant([
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]],

      ], dtype=tf.int32)
      resolution = tf.constant(4)
      out, ind, cnt = avg_voxelize_forward(features, coords, resolution)
    print(f"out={out}")
    print(f"ind={ind}")
    print(f"cnt={cnt}")


if __name__ == "__main__":
  tf.test.main()
