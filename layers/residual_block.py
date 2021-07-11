import tensorflow as tf
from utils.spectral_normalizer import spectral_normalizer
from layers.conditional_batch_normalization import ConditionalBatchNormalization


class ResidualBlock(tf.layers.Layer):
    '''Residual Block Layer
    '''

    def __init__(self,
                 out_c=None,
                 hidden_c=None,
                 ksize=3,
                 stride=1,
                 activation=tf.nn.relu,
                 is_use_bn=True,
                 upsampling=False,
                 downsampling=False,
                 category=0,
                 is_use_sn=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ResidualBlock, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.out_c = out_c
        self.hidden_c = hidden_c
        self.ksize = ksize
        self.stride = stride
        self.activation = activation
        self.is_use_bn = is_use_bn
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.category = category
        self.is_use_sn = is_use_sn
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        self.in_c = input_shape[-1]
        if self.out_c is None:
            self.out_c = self.in_c
        if self.hidden_c is None:
            self.hidden_c = self.in_c
        self.is_shortcut_learn = (self.in_c != self.out_c) or self.upsampling or self.downsampling

        self.conv1 = tf.layers.Conv2D(self.hidden_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_u = None
        self._layers.append(self.conv1)

        self.conv2 = tf.layers.Conv2D(self.out_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_u = None
        self._layers.append(self.conv2)

        if self.is_use_bn:
            if self.category:
                self.bn1 = ConditionalBatchNormalization(self.category)
                self._layers.append(self.bn1)
                self.bn2 = ConditionalBatchNormalization(self.category)
                self._layers.append(self.bn2)
            else:
                self.bn1 = tf.layers.BatchNormalization()
                self._layers.append(self.bn1)
                self.bn2 = tf.layers.BatchNormalization()
                self._layers.append(self.bn2)

        if self.is_shortcut_learn:
            self.conv_shortcut = tf.layers.Conv2D(self.out_c,
                                                  1,
                                                  strides=(1, 1),
                                                  padding='SAME',
                                                  use_bias=False,
                                                  activation=None,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.conv_shortcut_u = None
            self._layers.append(self.conv_shortcut)

    @property
    def variables(self):
        vars = []
        for l in self._layers:
            vars += l.variables
        return vars
    
    @property
    def trainable_variables(self):
        vars = []
        for l in self._layers:
            vars += l.trainable_variables
        return vars

    def _upsample(self, var):
        height = var.shape[1]
        width = var.shape[2]
        return tf.image.resize_images(var,
                                      (height * 2, width * 2),
                                      method=tf.image.ResizeMethod.BILINEAR)

    def _downsample(self, var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')

    def call(self, inputs, labels=None, training = True):
        out = inputs
        if self.is_use_bn:
            out = self.bn1(out, training = training) if labels is None else self.bn1(out, labels=labels, training = training)
        out = self.activation(out)
        if self.upsampling:
            out = self._upsample(out)

        if self.is_use_sn:
            if not self.conv1.built:
                self.conv1.build(out.shape)
            with tf.variable_scope("conv1"):
                self.conv1_u = tf.get_variable(
                                    name="u",
                                    shape=(1, self.conv1.kernel.shape[-1]),
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=False)
            kernel_mat = tf.reshape(tf.transpose(self.conv1.kernel, (3, 2, 1, 0)), (self.conv1.kernel.shape[-1], -1))
            sigma, new_u = spectral_normalizer(kernel_mat, self.conv1_u)
            with tf.control_dependencies([self.conv1.kernel.assign(self.conv1.kernel / sigma), self.conv1_u.assign(new_u)]):
                out = self.conv1(out)
        else:
            out = self.conv1(out)

        if self.is_use_bn:
            out = self.bn2(out, training = training) if labels is None else self.bn2(out, labels=labels, training = training)
        out = self.activation(out)

        if self.is_use_sn:
            if not self.conv2.built:
                self.conv2.build(out.shape)
            with tf.variable_scope("conv2"):
                self.conv2_u = tf.get_variable(
                                    name="u",
                                    shape=(1, self.conv2.kernel.shape[-1]),
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=False)
            kernel_mat = tf.reshape(tf.transpose(self.conv2.kernel, (3, 2, 1, 0)), (self.conv2.kernel.shape[-1], -1))
            sigma, new_u = spectral_normalizer(kernel_mat, self.conv2_u)
            with tf.control_dependencies([self.conv2.kernel.assign(self.conv2.kernel / sigma), self.conv2_u.assign(new_u)]):
                out = self.conv2(out)
        else:
            out = self.conv2(out)

        if self.downsampling:
            out = self._downsample(out)

        if self.is_shortcut_learn:
            control_flow = tf.control_dependencies([])
            if self.is_use_sn:
                if not self.conv_shortcut.built:
                    self.conv_shortcut.build(inputs.shape)
                with tf.variable_scope("conv_shortcut"):
                    self.conv_shortcut_u = tf.get_variable(
                                        name="u",
                                        shape=(1, self.conv_shortcut.kernel.shape[-1]),
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        trainable=False)
                kernel_mat = tf.reshape(tf.transpose(self.conv_shortcut.kernel, (3, 2, 1, 0)), (self.conv_shortcut.kernel.shape[-1], -1))
                sigma, new_u = spectral_normalizer(kernel_mat, self.conv_shortcut_u)
                control_flow = tf.control_dependencies([self.conv_shortcut.kernel.assign(self.conv_shortcut.kernel / sigma), self.conv_shortcut_u.assign(new_u)])

            with control_flow:
                if self.upsampling:
                    x = self.conv_shortcut(self._upsample(inputs))
                elif self.downsampling:
                    x = self._downsample(self.conv_shortcut(inputs))
                else:
                    x = self.conv_shortcut(inputs)
        else:
            x = inputs
        return out + x

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        height = input_shape[1]
        width = input_shape[2]
        if self.upsampling:
            height *= 2
            width *= 2
        return tf.TensorShape(
          [input_shape[0], height, width, self.out_c])

class OptimizedBlock(tf.layers.Layer):
    '''Optimized Residual Block Layer for discriminator
    '''

    def __init__(self,
                 out_c=None,
                 ksize=3,
                 stride=1,
                 activation=tf.nn.relu,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(OptimizedBlock, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.out_c = out_c
        self.ksize = ksize
        self.stride = stride
        self.activation = activation
        self._layers = []

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        self.in_c = input_shape[-1]
        if self.out_c is None:
            self.out_c = self.in_c

        self.is_shortcut_learn = True

        self.conv1 = tf.layers.Conv2D(self.out_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_u = None
        self._layers.append(self.conv1)

        self.conv2 = tf.layers.Conv2D(self.out_c,
                                      self.ksize,
                                      strides=(self.stride, self.stride),
                                      padding='SAME',
                                      use_bias=False,
                                      activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_u = None
        self._layers.append(self.conv2)

        if self.is_shortcut_learn:
            self.conv_shortcut = tf.layers.Conv2D(self.out_c,
                                                  1,
                                                  strides=(1, 1),
                                                  padding='SAME',
                                                  use_bias=False,
                                                  activation=None,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.conv_shortcut_u = None
            self._layers.append(self.conv_shortcut)

    @property
    def variables(self):
        vars = []
        for l in self._layers:
            vars += l.variables
        return vars
        
    @property
    def trainable_variables(self):
        vars = []
        for l in self._layers:
            vars += l.trainable_variables
        return vars

    def _downsample(self, var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')

    def call(self, inputs):
        out = inputs

        if not self.conv1.built:
            self.conv1.build(out.shape)
        with tf.variable_scope("conv1"):
            self.conv1_u = tf.get_variable(
                                name="u",
                                shape=(1, self.conv1.kernel.shape[-1]),
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=False)
        kernel_mat = tf.reshape(tf.transpose(self.conv1.kernel, (3, 2, 1, 0)), (self.conv1.kernel.shape[-1], -1))
        sigma, new_u = spectral_normalizer(kernel_mat, self.conv1_u)
        with tf.control_dependencies([self.conv1.kernel.assign(self.conv1.kernel / sigma), self.conv1_u.assign(new_u)]):
            out = self.conv1(out)

        out = self.activation(out)

        if not self.conv2.built:
            self.conv2.build(out.shape)
        with tf.variable_scope("conv2"):
            self.conv2_u = tf.get_variable(
                                name="u",
                                shape=(1, self.conv2.kernel.shape[-1]),
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=False)
        kernel_mat = tf.reshape(tf.transpose(self.conv2.kernel, (3, 2, 1, 0)), (self.conv2.kernel.shape[-1], -1))
        sigma, new_u = spectral_normalizer(kernel_mat, self.conv2_u)
        with tf.control_dependencies([self.conv2.kernel.assign(self.conv2.kernel / sigma), self.conv2_u.assign(new_u)]):
            out = self.conv2(out)

        out = self._downsample(out)

        if self.is_shortcut_learn:
            control_flow = tf.control_dependencies([])

            if not self.conv_shortcut.built:
                self.conv_shortcut.build(inputs.shape)
            with tf.variable_scope("conv_shortcut"):
                self.conv_shortcut_u = tf.get_variable(
                                    name="u",
                                    shape=(1, self.conv_shortcut.kernel.shape[-1]),
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=False)
            kernel_mat = tf.reshape(tf.transpose(self.conv_shortcut.kernel, (3, 2, 1, 0)), (self.conv_shortcut.kernel.shape[-1], -1))
            sigma, new_u = spectral_normalizer(kernel_mat, self.conv_shortcut_u)
            control_flow = tf.control_dependencies([self.conv_shortcut.kernel.assign(self.conv_shortcut.kernel / sigma), self.conv_shortcut_u.assign(new_u)])

            with control_flow:
                x = self.conv_shortcut(self._downsample(inputs))

        else:
            x = inputs
        return out + x

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        height = input_shape[1]
        width = input_shape[2]
        if self.upsampling:
            height *= 2
            width *= 2
        return tf.TensorShape(
          [input_shape[0], height, width, self.out_c])
