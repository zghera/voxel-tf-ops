"""Tests for avg_vox ops."""
import tensorflow as tf

try:
  from avg_vox.python.ops import avg_voxelize_forward, avg_voxelize_backward
except ImportError:
  from avg_vox_ops import avg_voxelize_forward, avg_voxelize_backward


class AvgVoxTest(tf.test.TestCase):
  B = 2
  C = 5
  N = 4
  R = 4

  # Note: We only see 1, and 2, 3 in `out` because of the "clipping" that was
  #       added in GridStatsKernel. More specifically, the equation for
  #       calculating a point's voxel index may give a voxel index greater
  #       than R**3 (point (4,4,4) in this case). If that occurs, we do not
  #       increment the element in `cnt` corresponding to that voxel. Thus, we
  #       do not see the corresponding feature for that point in `out`.
  def test_avg_voxelize_forward(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      # [B, C, R**3] = [2, 5, 64]
      expected_out = tf.constant([[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,],
        ]], dtype=tf.float32)
      expected_out = tf.repeat(tf.repeat(expected_out, 5, axis=1), 2, axis=0)

      # [B, R**3] = [2, 64]
      expected_cnt = tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      ], dtype=tf.int32)
      expected_cnt = tf.repeat(expected_cnt, 2, axis=0)

      # [B, N] = [2, 4]
      expected_ind = tf.constant([
        [21, 42, 63, 84], [21, 42, 63, 84]
      ], dtype=tf.int32)


      # [B, C, N] = [2, 5, 4]
      features = tf.constant([[[1, 2, 3, 4]]], dtype=tf.float32)
      features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)

      # [B, 3, N] = [2, 3, 4]
      coords = tf.constant([[[1, 2, 3, 4]]], dtype=tf.int32)
      coords = tf.repeat(tf.repeat(coords, 3, axis=1), 2, axis=0)

      resolution = tf.constant(self.R)


      out, ind, cnt = avg_voxelize_forward(features, coords, resolution)

      self.assertAllClose(out, expected_out)
      self.assertAllClose(cnt, expected_cnt)
      self.assertAllClose(ind, expected_ind)
    # fmt: on

  def test_avg_voxelize_gradients(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      # [B, C, N] = [2, 5, 4]
      expected_features_grad = tf.constant([[[0, 0.003125, 0.00625, 0]]])
      expected_features_grad = tf.repeat(
          tf.repeat(expected_features_grad, 5, axis=1), 2, axis=0)


      # [B, C, N] = [2, 5, 4]
      features = tf.constant([[[1, 2, 3, 4]]], dtype=tf.float32)
      features = tf.repeat(tf.repeat(features, 5, axis=1), 2, axis=0)

      # [B, 3, N] = [2, 3, 4]
      coords = tf.constant([[[1, 2, 3, 4]]], dtype=tf.int32)
      coords = tf.repeat(tf.repeat(coords, 3, axis=1), 2, axis=0)

      resolution = tf.constant(self.R)


      out, ind, cnt = avg_voxelize_forward(features, coords, resolution)

      mse = tf.keras.losses.MeanSquaredError()
      label = tf.ones(shape=[self.B, self.C, self.R**3])
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(out)
        loss = mse(label, out)
        dL_dout = tape.gradient(loss, out)

      features_grad = avg_voxelize_backward(dL_dout, ind, cnt)

      self.assertAllClose(features_grad, expected_features_grad)
    # fmt: on


if __name__ == "__main__":
  tf.test.main()
