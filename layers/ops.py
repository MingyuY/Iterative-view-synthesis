import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.image import dense_image_warp

#weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_init = xavier_initializer()
# weight_init = variance_scaling_initializer()


# weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
weight_regularizer = None

# pad = (k-1) // 2 = SAME !
# output = ( input - k + 1 + 2p ) // s

def conv_2d(x, channels, kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, name='conv_0'):
    with tf.variable_scope(name):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)
        return x

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    print ("current", shape[:-1], fan_in)
    std = gain / np.sqrt(fan_in) # He init

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        # return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
        return tf.get_variable('weight', shape=shape, initializer=weight_init)
    else:
        # return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
        return tf.get_variable('weight', shape=shape, initializer=weight_init)

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, gain=np.sqrt(2), use_wscale=False, sn=False, padding='SAME',
           name="conv2d", with_w=False):
    with tf.variable_scope(name):

        w = get_weight([k_h, k_w, input_.shape[-1].value, output_dim], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, input_.dtype)

        if padding == 'Other':
            padding = 'VALID'
            input_ = tf.pad(input_, [[0,0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        if sn:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


        if with_w:
            return conv, w, biases

        else:
            return conv

def downscale2d(x, k=2):
    # avgpool wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)

        return x

def max_pooling(x, kernel=2, stride=2) :
    return tf.layers.max_pooling2d(x, pool_size=kernel, strides=stride)

def avg_pooling(x, kernel=2, stride=2) :
    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride)

def global_avg_pooling(x):
    """
    Incoming Tensor shape must be 4-D
    """
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def fully_connected(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x


def flatten(x) :
    return tf.contrib.layers.flatten(x)


#def lrelu(x, alpha=0.2):
#    # pytorch alpha is 0.01
#    return tf.nn.leaky_relu(x, alpha)

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def swish(x):
    return x * sigmoid(x)

def discriminator_loss(real, fake):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
    loss = real_loss + fake_loss

    return loss


def generator_loss(fake):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss



def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True,
                                        is_training=is_training, scope=scope)

    # return tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-05, center=True, scale=True, training=is_training, name=scope)



def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def group_norm(x, G=32, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C],
                               initializer=tf.constant_initializer(0.0))
        # gamma = tf.reshape(gamma, [1, 1, 1, C])
        # beta = tf.reshape(beta, [1, 1, 1, C])

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    # loss = tf.reduce_mean(tf.square(x - y))
    if len(x.get_shape().as_list()) == 4:
        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis = [1,2,3]))
    elif len(x.get_shape().as_list()) == 2:
        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis= [1]))
    return loss

def myinstance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[1,2], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

def apply_bias(x, name=None, lrmul=1):
    b = tf.get_variable(name or 'bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, 1, 1, -1])


def apply_noise(inputs, noise_var=None, name=None, random_noise=True):
    assert len(inputs.shape) == 4
    input_shape = inputs.shape.as_list()

    with tf.variable_scope(name or 'Noise'):
        if noise_var is None and random_noise:
            noise_input = tf.random_normal([input_shape[0], input_shape[1], input_shape[2], 1], dtype=inputs.dtype)
        else:
            noise_input = tf.cast(noise_var, inputs.dtype)
        weight = tf.get_variable('weight', shape = [input_shape[-1]],  dtype=inputs.dtype, initializer=tf.initializers.zeros())
        noise = noise_input * tf.reshape(weight, [1,1,1,-1])
        x = inputs + noise
        return x

def style_mod(x, dlatent, use_bias=False, name=None):
    shape = x.shape.as_list()
    with tf.variable_scope(name or 'StyleMod'):
        style =  tf.layers.dense(dlatent, units=shape[-1] * 2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        # style = tf.reshape(style, [-1, 2, shape[-1]] + [1] * (len(shape) - 2))
        style = tf.reshape(style, [-1, 2, 1, 1, shape[-1]])
        # return x * tf.exp(style[:, 0] + 1) + style[:, 1]
        return x * (style[:, 0] + 1) + style[:, 1]

def style_mod_rev(x, dlatent, use_bias=False, name=None):
    shape = x.shape.as_list()
    with tf.variable_scope(name or 'StyleMod'):
        style = tf.layers.dense(dlatent, units=shape[-1] * 2, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer, use_bias=use_bias)
        # style = tf.reshape(style, [-1, 2, shape[-1]] + [1] * (len(shape) - 2))
        style = tf.reshape(style, [-1, 2, 1, 1, shape[-1]])
        # greater = tf.greater(1.0 + style[:, 0], 0)
        # sign = tf.cast(greater, tf.float32) * 2 - 1
        # # sigma = tf.cond(, lambda:1.0 + style[:, 0] + 1e-6, lambda:1.0 + style[:, 0] - 1e-6)
        # sigma = 1.0 + style[:, 0] + 1e-6 * sign
        sigma = tf.exp(- 1.0 - style[:, 0])

        return (x - style[:, 1])*sigma

def Pixl_Norm(x, eps=1e-8):
    if len(x.shape) > 2:
        axis_ = 3
    if len(x.shape) == 3:
        axis_ = 2
    else:
        axis_ = 1
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis_, keep_dims=True) + eps)

def MinibatchstateConcat(input, averaging='all'):
    s = input.shape
    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")

    vals = tf.tile(vals, multiples=[s[0], s[1], s[2], 1])
    return tf.concat([input, vals], axis=3)

def lerp(a, b, t):
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t

def flow_warping(x, flow):
    x_shape = x.shape.as_list()
    flow_shape = flow.shape.as_list()
    assert x_shape[:-1]==flow_shape[:-1]
    assert flow_shape[-1]==2
    return dense_image_warp(x, flow)