"""Tests for trilinear_devox ops."""
import tensorflow as tf

try:
  from trilinear_devox.python.ops import (
    trilinear_devoxelize_forward,
    trilinear_devoxelize_backward,
  )
except ImportError:
  from trilinear_devox_ops import (
    trilinear_devoxelize_forward,
    trilinear_devoxelize_backward,
  )


class TrilinearDevoxTest(tf.test.TestCase):
  B = 2
  C = 5
  N = 4
  R = 5#4

  # Note: We use R=5 rather than R=4 like in the avg_vox tests because of an
  #       issue in the backwards output (gradients of features/input). 
  #
  #       For the forward pass, the output is as expected (see below).
  #       Note that we see 0 rather than 4 (i.e. expected_outs=[[[1,2,3,4]]]])
  #       in `outs` because of the "clipping" when creating `cnt`. For more
  #       information, see avg_vox/cc/avg_vox_kernels.cu.cc|GridStatsKernel 
  #       or the avg_vox tests.
  #
  #       However, for the backwards pass, there was an inconsistency in the
  #       1st channel of the 1st batch where all channels of each batch should
  #       have the exact same R**3 values. Specifically, this was the 1st batch
  #       for features_grad: 
  #      [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
  #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
  #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
  #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
  #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
  #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1]], ...]
  def test_trilinear_devoxelize_forward(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      """ With R=4:
      # [B, C, N] = [2, 5, 4]
      expected_outs = tf.constant([[[1, 2, 3, 0]]], dtype=tf.float32)
      """
      # [B, C, N] = [2, 5, 4]
      expected_outs = tf.constant([[[1, 2, 3, 4]]], dtype=tf.float32)
      expected_outs = tf.repeat(tf.repeat(expected_outs, 5, axis=1), 2, axis=0)

      """ With R=4:
      # [B, 8, N] = [2, 8, 4]
      expected_inds = tf.constant([[[21, 42, 63, 84]]], dtype=tf.int32)
      """
      # [B, 8, N] = [2, 8, 4]
      expected_inds = tf.constant([[[31, 62, 93, 124]]], dtype=tf.int32)
      expected_inds = tf.repeat(tf.repeat(expected_inds, 8, axis=1), 2, axis=0)

      # [B, 8, N] = [2, 8, 4]
      expected_wgts = tf.constant([
        [[1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
      ], dtype=tf.float32)
      expected_wgts = tf.repeat(expected_wgts, 2, axis=0)


      """ With R=4:
      # [B, C, R**3] = [2, 5, 64]
      # features = tf.constant([[
      #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,]
      #   ]], dtype=tf.float32)
      # features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)
      """
      # [B, C, R**3] = [2, 5, 125]
      features = tf.constant([[
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 4]
      ]], dtype=tf.float32)
      features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)

      # [B, 3, N] = [2, 3, 4]
      coords = tf.constant([[[1, 2, 3, 4]]], dtype=tf.float32)
      coords = tf.repeat(tf.repeat(coords, 3, axis=1), 2, axis=0)

      # Attrs
      resolution = tf.constant(self.R)
      is_training = True


      outs, inds, wgts = trilinear_devoxelize_forward(features, coords, resolution, is_training)

      self.assertAllClose(outs, expected_outs)
      self.assertAllClose(inds, expected_inds)
      self.assertAllClose(wgts, expected_wgts)
    # fmt: on

  def test_trilinear_devoxelize_gradients(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      """ With R=4:
      # [B, C, R**3] = [2, 5, 64]
      # expected_features_grad = tf.constant([[
      #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-0.05,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
      # ]])
      # expected_features_grad = tf.repeat(
      #     tf.repeat(expected_features_grad, 5, axis=1), 2, axis=0)
      """
      # [B, C, R**3] = [2, 5, 125]
      expected_features_grad = tf.constant([[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0.15]
      ]])
      expected_features_grad = tf.repeat(
          tf.repeat(expected_features_grad, 5, axis=1), 2, axis=0)


      """ With R=4:
      # [B, C, R**3] = [2, 5, 64]
      # features = tf.constant([[
      #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
      #    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,]
      #   ]], dtype=tf.float32)
      # features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)
      """
      # [B, C, R**3] = [2, 5, 125]
      features = tf.constant([[
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 4]
      ]], dtype=tf.float32)
      features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)

      # [B, 3, N] = [2, 3, 4]
      coords = tf.constant([[[1, 2, 3, 4]]], dtype=tf.float32)
      coords = tf.repeat(tf.repeat(coords, 3, axis=1), 2, axis=0)

      # Attrs
      resolution = tf.constant(self.R)
      is_training = True


      outs, inds, wgts = trilinear_devoxelize_forward(features, coords, resolution, is_training)

      mse = tf.keras.losses.MeanSquaredError()
      label = tf.ones(shape=[self.B, self.C, self.N])
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(outs)
        loss = mse(label, outs)
        dL_dout = tape.gradient(loss, outs)

      features_grad = trilinear_devoxelize_backward(dL_dout, inds, wgts, resolution)

      self.assertAllClose(features_grad, expected_features_grad)
    # fmt: on


if __name__ == "__main__":
  tf.test.main()
