import tensorflow as tf
from utils.spectral_normalizer import spectral_normalizer
from layers.conditional_batch_normalization import ConditionalBatchNormalization
import tensorflow.contrib.layers as tcl
from models.ops import *

class ResidualBlock():
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
                 name='ResBlock'):
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
        self.name = name

    def _upsample(self, var):
        height = var.shape.as_list()[1]
        width = var.shape.as_list()[2]
        return tf.image.resize_images(var,
                                      (height * 2, width * 2),
                                      method=tf.image.ResizeMethod.BILINEAR)

    def _downsample(self, var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME', data_format='channels_last')

    def __call__(self, inputs, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = inputs
            input_shape = inputs.shape.as_list()
            self.in_c = input_shape[-1]
            if self.out_c is None:
                self.out_c = self.in_c
            if self.hidden_c is None:
                self.hidden_c = self.in_c
            self.is_shortcut_learn = (self.in_c != self.out_c) or self.upsampling or self.downsampling
            
            if self.is_use_bn:
                if self.category:
                    bn1 = ConditionalBatchNormalization(self.category)
                    out = bn1(out, labels=labels, training = training)
                else:
                    out = tcl.batch_norm(out, scale = True, is_training = training)
            out = self.activation(out)
            if self.upsampling:
                out = self._upsample(out)
    
            if self.is_use_sn:
                out = conv(out, channels=self.hidden_c, kernel=self.ksize, stride=self.stride, pad=1, use_bias=False, sn=True, scope='conv_1')
            else:
                out = tcl.conv2d(out, num_outputs=self.hidden_c, kernel_size=self.ksize, stride=self.stride, 
                                 padding='SAME', activation_fn=None, biases_initializer=None)
    
            if self.is_use_bn:
                if self.category:
                    bn2 = ConditionalBatchNormalization(self.category)
                    out = bn2(out, labels=labels, training = training)
                else:
                    out = tcl.batch_norm(out, scale = True, is_training = training)
            out = self.activation(out)
    
            if self.is_use_sn:
                out = conv(out, channels=self.out_c, kernel=self.ksize, stride=self.stride, pad=1, use_bias=False, sn=True, scope='conv_2')
            else:
                out = tcl.conv2d(out, num_outputs=self.out_c, kernel_size=self.ksize, stride=self.stride,
                                 padding='SAME', activation_fn=None, biases_initializer=None)
    
            if self.downsampling:
                out = self._downsample(out)
    
            if self.is_shortcut_learn:
                if self.is_use_sn:
                    if self.upsampling:
                        x = conv(self._upsample(inputs), channels=self.out_c, kernel=1, stride=1, pad=0, 
                                 use_bias=False, sn=True, scope='conv_short')
                    elif self.downsampling:
                        x = self._downsample(conv(inputs, channels=self.out_c, kernel=1, stride=1, pad=0, 
                                 use_bias=False, sn=True, scope='conv_short'))
                    else:
                        x = conv(inputs, channels=self.out_c, kernel=1, stride=1, pad=0, 
                                 use_bias=False, sn=True, scope='conv_short')
                else:
                    if self.upsampling:
                        x = tcl.conv2d(self._upsample(inputs), num_outputs=self.out_c, kernel_size=1, stride=1, 
                                 padding='SAME', activation_fn=None, biases_initializer=None)
                    elif self.downsampling:
                        x = self._downsample(tcl.conv2d(inputs, num_outputs=self.out_c, kernel_size=1, stride=1, 
                                 padding='SAME', activation_fn=None, biases_initializer=None))
                    else:
                        x = tcl.conv2d(inputs, num_outputs=self.out_c, kernel_size=1, stride=1, 
                                 padding='SAME', activation_fn=None, biases_initializer=None)
            else:
                x = inputs
            return out + x
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class OptimizedBlock():
    '''Optimized Residual Block Layer for discriminator
    '''

    def __init__(self,
                 out_c=None,
                 ksize=3,
                 stride=1,
                 activation=tf.nn.relu,
                 trainable=True,
                 name="OptimizeBlock"):
        self.out_c = out_c
        self.ksize = ksize
        self.stride = stride
        self.activation = activation
        self.name = name

    def _downsample(self, var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME', data_format='channels_last')

    def __call__(self, inputs, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
                
            input_shape = inputs.shape.as_list()
            self.in_c = input_shape[-1]
            if self.out_c is None:
                self.out_c = self.in_c
            self.is_shortcut_learn = True
            
            out = inputs
            out = conv(out, channels=self.out_c, kernel=self.ksize, stride=self.stride, pad=1, 
                                 use_bias=False, sn=True, scope='conv_1')
            out = self.activation(out)
            
            out = conv(out, channels=self.out_c, kernel=self.ksize, stride=self.stride, pad=1, 
                                 use_bias=False, sn=True, scope='conv_2')
            out = self._downsample(out)
    
            if self.is_shortcut_learn:
                x = conv(self._downsample(inputs), channels=self.out_c, kernel=1, stride=1, pad=0, 
                                 use_bias=False, sn=True, scope='conv_short')
            else:
                x = inputs
            return out + x
            
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
