import tensorflow as tf


def spectral_normalizer(W, u):
    v = tf.nn.l2_normalize(tf.matmul(u, W))
    _u = tf.nn.l2_normalize(tf.matmul(v, W, transpose_b=True))
    sigma = tf.matmul(tf.matmul(_u, W), v, transpose_b=True)
    sigma = tf.reduce_sum(sigma)
    return sigma, _u
