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
  from avg_vox.python.ops import avg_voxelize_forward, avg_voxelize_backward
except ImportError:
  from avg_vox_ops import avg_voxelize_forward, avg_voxelize_backward


class AvgVoxTest(tf.test.TestCase):
  def test_avg_voxelize_forward(self):
    # fmt: off
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

      expected_out = tf.constant([
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,]]
        ], dtype=tf.float32)
      expected_out = tf.repeat(expected_out, 2, axis=0)
      expected_cnt = tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
      ], dtype=tf.int32)
      expected_ind = tf.constant([
        [21, 42, 63, 84], [21, 42, 63, 84]
      ], dtype=tf.int32)

      self.assertAllClose(out, expected_out)
      self.assertAllClose(cnt, expected_cnt)
      self.assertAllClose(ind, expected_ind)
    # fmt: on

  def test_avg_voxelize_gradients(self):
    # fmt: off
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

      mse = tf.keras.losses.MeanSquaredError()
      label = tf.ones(shape=[2, 5, 4**3])
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(out)
        loss = mse(label, out)
        dL_dout = tape.gradient(loss, out)

      features_grad = avg_voxelize_backward(dL_dout, ind, cnt)
      expected_features_grad = tf.constant([
        [[0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0]],
        [[0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0],
         [0,  0.003125, 0.00625,  0]]
      ])

      self.assertAllClose(features_grad, expected_features_grad)
    # fmt: on

if __name__ == "__main__":
  tf.test.main()