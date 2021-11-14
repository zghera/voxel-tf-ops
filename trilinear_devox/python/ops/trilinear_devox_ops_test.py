"""Tests for trilinear_devox ops."""
import tensorflow as tf

try:
  from trilinear_devox.python.ops import (
    trilinear_devoxelize_forward,
    trilinear_devoxelize_backward,
  )
except ImportError:
  from trilinear_devox import (
    trilinear_devoxelize_forward,
    trilinear_devoxelize_backward,
  )


class TrilinearDevoxTest(tf.test.TestCase):
  B = 2
  C = 5
  N = 4
  R = 4

  def test_trilinear_devoxelize_forward(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      pass
    # fmt: on

  def test_trilinear_devoxelize_gradients(self):
    # fmt: off
    with tf.device("/device:GPU:0"):
      pass
    # fmt: on


if __name__ == "__main__":
  tf.test.main()
