import tensorflow as tf
#import tensorflow.contrib.distribute as distribute_lib
import tensorflow.contrib.util as utils


class ConditionalBatchNormalization(tf.layers.Layer):
    """Conditional Batch Normalization
  """

    def __init__(self,
                 category,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=tf.initializers.zeros(),
                 gamma_initializer=tf.initializers.ones(),
                 moving_mean_initializer=tf.initializers.zeros(),
                 moving_variance_initializer=tf.initializers.ones(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ConditionalBatchNormalization, self).__init__(
            name=name, trainable=trainable, **kwargs)
        if isinstance(axis, list):
            self.axis = axis[:]
        else:
            self.axis = axis
        self.category = category
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint

        self._bessels_correction_test_only = True

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        if not isinstance(self.axis, list):
            raise TypeError(
                'axis must be int or list, type given: %s' % type(self.axis))

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            param_dtype = tf.float32
        else:
            param_dtype = self.dtype or tf.float32

        axis_to_dim = {x: input_shape[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError(
                    'Input has undefined `axis` dimension. Input shape: ',
                    input_shape)
        self.input_spec = tf.layers.InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = [list(axis_to_dim.values())[0]]
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
            ]

        if self.scale:
            self.gamma = self.add_variable(
                name='gamma',
                shape=[self.category] + param_shape,
                dtype=param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_variable(
                name='beta',
                shape=[self.category] + param_shape,
                dtype=param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True)
        else:
            self.beta = None

        # Disable variable partitioning when creating the moving mean and variance
        try:
            if self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_variable(
                name='moving_mean',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.moving_mean_initializer,
                trainable=False)

            self.moving_variance = self.add_variable(
                name='moving_variance',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.moving_variance_initializer,
                trainable=False)

        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _assign_moving_average(self, variable, value, momentum):
        with tf.name_scope(None, 'AssignMovingAvg',
                           [variable, value, momentum]) as scope:
            decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - value) * decay
            return tf.assign_sub(variable, update_delta, name=scope)

    def call(self, inputs, labels=None, training=False):
        '''
        Args:
            labels: shape is [batch_size, 1]. value is in category.
        '''
        in_eager_mode = tf.contrib.eager.executing_eagerly()

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis
                          ]  # axis 

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.get_shape()) != ndims
                    and reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale = _broadcast(tf.nn.embedding_lookup(self.gamma, labels))
        scale = tf.expand_dims(scale, 1)
        offset = _broadcast(tf.nn.embedding_lookup(self.beta, labels))
        offset = tf.expand_dims(offset, 1)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        if training is None:
            training_value = True
        else:
            training_value = training
        if training_value is not False:

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = len(self.axis) > 1
            mean, variance = tf.nn.moments(
                inputs, reduction_axes, keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf.cond(training, lambda: mean,
                                    lambda: moving_mean)
            variance = tf.cond(training, lambda: variance,
                                        lambda: moving_variance)

            new_mean, new_variance = mean, variance

            def _do_update(var, value):
                if in_eager_mode and not self.trainable:
                    return

                return self._assign_moving_average(var, value, self.momentum)

            mean_update = tf.cond(
                training, lambda: _do_update(self.moving_mean, new_mean),
                lambda: self.moving_mean)
            variance_update = tf.cond(
                training,
                lambda: _do_update(self.moving_variance, new_variance),
                lambda: self.moving_variance)
            if not tf.contrib.eager.executing_eagerly():
                self.add_update(mean_update, inputs=inputs)
                self.add_update(variance_update, inputs=inputs)

        else:
            mean, variance = self.moving_mean, self.moving_variance

        outputs = tf.nn.batch_normalization(
            inputs,
            _broadcast(mean),
            _broadcast(variance),
            offset,
            scale,
            self.epsilon)
        outputs.set_shape(input_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
