import tensorflow as tf
import numpy as np


N = 2
B = 3
C = 5
D = 4

graph = tf.Graph()

with graph.as_default():
    x = tf.random_uniform(dtype=tf.int32, minval=0, maxval=10, shape=[N, B, C])
    y = tf.random_uniform(dtype=tf.int32, minval=0, maxval=10, shape=[N, C, D])
    z = tf.matmul(x, y)

    # x2 = tf.expand_dims(x, 2)
    # x3 = tf.tile(x2, [1, 1, C, 1])

    # y2 = tf.expand_dims(y, 1)
    # y3 = tf.tile(y2, [1, B, 1, 1])

    # w = tf.concat([x3, y3, x3 * y3], axis=-1)

    # x_mask = tf.sequence_mask([1, 2], B)
    # q_mask = tf.sequence_mask([3, 1], C)
    # xq_mask = tf.tile(tf.expand_dims(x_mask, axis=-1), [1, 1, C])
    # xq_mask2 = tf.tile(tf.expand_dims(q_mask, axis=1), [1, B, 1])
    # xq_mask3 = tf.cast(xq_mask, tf.float32) * tf.cast(xq_mask2, tf.float32)

with tf.Session(graph=graph) as session:
    # x, y, z, w, x_mask, q_mask, xq_mask3 = session.run([x, y, z, w, x_mask, q_mask, xq_mask3])
    x, y, z = session.run([x, y, z])
    print("x")
    print(x)
    print("y")
    print(y)
    print("z")
    print(z)
    assert z.shape == (N, B, D), z.shape
    # assert w.shape == (N, B, C, 3 * D)
    # print("w:")
    # print(w)
    # print("x_mask:\n%s" % x_mask)
    # print("q_mask:\n%s" % q_mask)
    # print("xq_mask3:\n%s" % xq_mask3)
