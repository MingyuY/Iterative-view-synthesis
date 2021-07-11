import tensorflow as tf
from layers.residual_block import ResidualBlock, OptimizedBlock
from utils.spectral_normalizer import spectral_normalizer
import tensorflow.contrib.layers as tcl
import numpy as np
from layers.ops import *

class LabelEmbeds40():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed64"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # out = labels
            # w_out1s=[]
            w_out2s=[]
            w_out3s=[]
            w_out4s=[]
            #w_out5s=[]
            for i in range(20):
                w_label = labels[:,i,:]
                w_out1 = tcl.fully_connected(w_label, self._channel, activation_fn=tf.nn.relu,
                                          normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})

                w_out1 = tf.reshape(w_out1, (-1, self._channel, 1, 1, 1))
                w_out2 = tcl.conv3d_transpose(w_out1, 1, [3,1,1], stride=[2,1,1], normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
                w_out3 = tcl.conv3d_transpose(w_out2, 1, [3,1,1], stride=[2,1,1], normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
                w_out4 = tcl.conv3d_transpose(w_out3, 1, [3,1,1], stride=[2,1,1], normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
                w_out5 = tcl.conv3d_transpose(w_out4, 1, [3,1,1], stride=1, normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
                # w_out1s.append(w_out1)
                w_out2s.append(w_out2)
                w_out3s.append(w_out3)
                w_out4s.append(w_out4)
#                w_out5s.append(w_out5)
#             out1 = tf.reshape(tf.concat(w_out1s, 1), (-1, 20, self._channel))
            out2 = tf.reshape(tf.concat(w_out2s, 1), (-1, 20, self._channel * 2))
            out3 = tf.reshape(tf.concat(w_out3s, 1), (-1, 20, self._channel * 4))
            out4 = tf.reshape(tf.concat(w_out4s, 1), (-1, 20, self._channel * 8))
            #out5 = tf.reshape(tf.concat(w_out5s, 1), (-1, 40, self._channel * 8))
            # out6 = tcl.fully_connected(out, self._channel*16, activation_fn=tf.nn.relu,
            #                           normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out7 = tcl.fully_connected(out, self._channel*32, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            return out2, out3, out4
#            return out1,out2
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_all32():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_all32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            inputs = tf.transpose(labels, (1,0,2))  # [40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]

            # bias1 = tf.get_variable("bias1", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias2 = tf.get_variable("bias2", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias3 = tf.get_variable("bias3", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias4 = tf.get_variable("bias4", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())

            # out1 = tf.nn.relu(tf.matmul(inputs, y_emb1) + bias1)  # [40,N,32]
            # out2 = tf.nn.relu(tf.matmul(out1, y_emb2) + bias2)
            # out3 = tf.nn.relu(tf.matmul(out2, y_emb3) + bias3)
            # out4 = tf.nn.relu(tf.matmul(out3, y_emb4) + bias4)
            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True, is_training=training)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True, is_training=training)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True, is_training=training)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True, is_training=training)
            # out4 = tf.matmul(out3, y_emb4)
            # out1 = tf.reshape(out1, (-1, self.y_num, self._channel))  # [N,40,32]
            # out2 = tf.reshape(out2, (-1, self.y_num, self._channel))
            # out3 = tf.reshape(out3, (-1, self.y_num, self._channel))
            # out4 = tf.reshape(out4, (-1, self.y_num, self._channel))
            out1 = tf.transpose(out1, (1,0,2))  # [N,40,32]
            out2 = tf.transpose(out2, (1,0,2))  # [N,40,32]
            out3 = tf.transpose(out3, (1,0,2))  # [N,40,32]
            out4 = tf.transpose(out4, (1,0,2))  # [N,40,32]

            return out1, out2, out3, out4
            # return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_lastshare():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            inputs = tf.transpose(labels, (1,0,2))  # [40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]

            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            out1 = tf.nn.leaky_relu(out1)
            # out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tf.nn.leaky_relu(out2)
            # out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tf.nn.leaky_relu(out3)
            # out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tf.nn.leaky_relu(out4)
            # out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)

            # out1 = tf.transpose(out1, (1,0,2))  # [N,40,32]
            # out2 = tf.transpose(out2, (1,0,2))  # [N,40,32]
            # out3 = tf.transpose(out3, (1,0,2))  # [N,40,32]
            out4 = tf.transpose(out4, (1,0,2))  # [N,40,32]

            return out4
            # return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_all32_long():
    """LabelEmbed
    """

    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_all32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim

    def __call__(self, batch_size, labels=None, training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            inputs = tf.transpose(labels, (1, 0, 2))  # [40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb5 = tf.get_variable("emb5", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb6 = tf.get_variable("emb6", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb7 = tf.get_variable("emb7", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb8 = tf.get_variable("emb8", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]

            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)
            out5 = tf.matmul(out4, y_emb5)
            out5 = tcl.batch_norm(out5, activation_fn=tf.nn.relu, scale=True)
            out6 = tf.matmul(out5, y_emb6)
            out6 = tcl.batch_norm(out6, activation_fn=tf.nn.relu, scale=True)
            out7 = tf.matmul(out6, y_emb7)
            out7 = tcl.batch_norm(out7, activation_fn=tf.nn.relu, scale=True)
            out8 = tf.matmul(out7, y_emb8)
            out8 = tcl.batch_norm(out8, activation_fn=tf.nn.relu, scale=True)
            # out4 = tf.matmul(out3, y_emb4)
            # out1 = tf.reshape(out1, (-1, self.y_num, self._channel))  # [N,40,32]
            # out2 = tf.reshape(out2, (-1, self.y_num, self._channel))
            # out3 = tf.reshape(out3, (-1, self.y_num, self._channel))
            # out4 = tf.reshape(out4, (-1, self.y_num, self._channel))
            out1 = tf.transpose(out1, (1, 0, 2))  # [N,40,32]
            out2 = tf.transpose(out2, (1, 0, 2))  # [N,40,32]
            out3 = tf.transpose(out3, (1, 0, 2))  # [N,40,32]
            out4 = tf.transpose(out4, (1, 0, 2))  # [N,40,32]
            out5 = tf.transpose(out5, (1, 0, 2))  # [N,40,32]
            out6 = tf.transpose(out6, (1, 0, 2))  # [N,40,32]
            out7 = tf.transpose(out7, (1, 0, 2))  # [N,40,32]
            out8 = tf.transpose(out8, (1, 0, 2))  # [N,40,32]

            return out1, out2, out3, out4, out5, out6, out7, out8
            # return out1

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
# class LabelEmbeds40_one32():
#     """LabelEmbed
#     """
#     def __init__(self,
#                  channel=32,
#                  activation=tf.nn.relu,
#                  category=0,
#                  y_num=40,
#                  y_dim=2,
#                  trainable=True,
#                  name="LabelEmbeds40_all32"):
#         self._channel = channel
#         self._activation = activation
#         self._category = category
#         self.name = name
#         self.y_num = y_num
#         self.y_dim = y_dim
#
#
#     def __call__(self, batch_size, labels=None, training = True, reuse = False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#
#             # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
#             # y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
#             #                          initializer=tcl.xavier_initializer())  # [40,2,32]
#             #
#             # out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
#             # out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
#             # out1 = tf.reshape(out1, (-1, self.y_num, self._channel))
#
#             w_out1s = []
#             for i in range(self.y_num):
#                 w_label = labels[:, i, :]  # [N,40,2]-># [N,2]
#                 w_out1 = tcl.fully_connected(w_label, self._channel, activation_fn=tf.nn.relu,
#                                              normalizer_fn=tcl.batch_norm,
#                                              normalizer_params={'scale': True, 'is_training': training})  # [N,32]
#
#                 w_out1s.append(w_out1)
#             out1 = tf.reshape(tf.concat(w_out1s, 1), (-1, self.y_num, self._channel))
#             # return out1, out2, out3, out4
#             # return out1, y_emb1, hout
#             return out1
#     @property
#     def vars(self):
#         return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_one32():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_all32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputs = tf.transpose(labels, (1,0,2))  # [N,40,2]->[40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]

            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True, is_training=training)
            out1 = tf.transpose(out1, (1,0,2))  # [40,N,2]->[N,40,2]

            # w_out1s = []
            # for i in range(self.y_num):
            #     w_label = labels[:, i, :]  # [N,40,2]-># [N,2]
            #     w_out1 = tcl.fully_connected(w_label, self._channel, activation_fn=tf.nn.relu,
            #                                  normalizer_fn=tcl.batch_norm,
            #                                  normalizer_params={'scale': True, 'is_training': training})  # [N,32]
            #
            #     w_out1s.append(w_out1)
            # out1 = tf.reshape(tf.concat(w_out1s, 1), (-1, self.y_num, self._channel))

            # return out1, out2, out3, out4
            # return out1, y_emb1
            return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_one32_false():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_all32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputs = tf.reshape(labels, (self.y_num, -1, 2))  # [N,40,2]->[40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]

            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out1 = tf.reshape(out1, (-1, self.y_num, self._channel))  # [40,N,2]->[N,40,2]

            return out1, y_emb1
            # return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
class Mapping_our():
    """LabelEmbed
    """
    def __init__(self,
                 channel=256,
                 activation=tf.nn.relu,
                 category=0,
                 f_num=1,
                 trainable=True,
                 name="Mapping"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.f_num = f_num

    def __call__(self, latent_size, batch_size, label_size=0, is_smooth=0, labels=None, reuse = tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse:
            #     scope.reuse_variables()

            mapping_layers = 4 
            # latent_z = np.random.randn(batch_size, latent_size)
            # x = tf.cast(latent_z, tf.float32)
            if label_size:
                w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
                y = tf.matmul(labels, tf.cast(w, tf.float32)) 

            # Normalize latents.
            # if norm_latent:
            x = Pixl_Norm(y)
            # x = labels
            if is_smooth:
                epsilon = 0.1
                smooth = np.random.choice(2, self._category).astype(np.float32)
                c_num = np.sum(smooth)
                smooth[np.where(smooth==1.0)] = epsilon / c_num
                x = (1.0 - epsilon) * labels + smooth
 
            for layer_idx in range(mapping_layers):
                fmaps = self._channel * self.f_num
                out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % layer_idx))  
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
class Mapping_disentangle():
    """LabelEmbed
    """
    def __init__(self,
                 channel=256,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="Mapping"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name 

    def __call__(self, latent_size, batch_size, label_size=0, is_smooth=0, labels=None, reuse = tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope: 

            mapping_layers = 4  
            if label_size:
                w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
                x = tf.matmul(labels, tf.cast(w, tf.float32))  
            x = Pixl_Norm(x) 
            for layer_idx in range(mapping_layers):
                fmaps = self._channel 
                out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % layer_idx))  
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



class Mapping():
    """LabelEmbed
    """
    def __init__(self,
                 channel=256,
                 activation=tf.nn.relu,
                 category=0,
                 f_num=1,
                 trainable=True,
                 name="Mapping"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.f_num = f_num

    def __call__(self, latent_size, batch_size, z_var=None, label_size=0, is_smooth=0, labels=None, reuse = tf.AUTO_REUSE, use_noise=True):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse:
            #     scope.reuse_variables()

            mapping_layers = 4
            
            # latent_z = np.random.randn(batch_size, latent_size)
            # x = tf.cast(latent_z, tf.float32)
            if label_size:
                w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
                y = tf.matmul(labels, tf.cast(w, tf.float32))
                if use_noise:
                    x = z_var
                    x = tf.concat([z_var, y], axis=1)
                else:
                    x= y

            # Normalize latents.
            # if norm_latent:
            x = Pixl_Norm(x)
            # x = labels
            if is_smooth:
                epsilon = 0.1
                smooth = np.random.choice(2, self._category).astype(np.float32)
                c_num = np.sum(smooth)
                smooth[np.where(smooth==1.0)] = epsilon / c_num
                x = (1.0 - epsilon) * labels + smooth
            # x =  (1.0 - epsilon) * labels + epsilon / (1.0 * self._category)


            # mapping layers
            for layer_idx in range(mapping_layers):
                fmaps = self._channel * self.f_num
                out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % layer_idx)) 
#            fmaps = self._channel * self.f_num
#            out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % 1))

            # if self.f_num == 1:
            #     out = out
            # else:
            #     out = tf.reshape(out, (-1, self._channel, self.f_num))
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



    
class DMapping():
    """LabelEmbed
    """
    def __init__(self,
                 channel=256,
                 activation=tf.nn.relu,
                 category=0,
                 f_num=1,
                 trainable=True,
                 name="Mapping"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.f_num = f_num

    def __call__(self, z_var, latent_size, batch_size, label_size=0, is_smooth=0, labels=None, reuse = tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse:
            #     scope.reuse_variables()

            mapping_layers = 4
            x = z_var
            # latent_z = np.random.randn(batch_size, latent_size)
            # x = tf.cast(latent_z, tf.float32)
            if label_size:
                w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
                y = tf.matmul(labels, tf.cast(w, tf.float32))
                x = tf.concat([z_var, y], axis=1)

            # Normalize latents.
            # if norm_latent:
            x = Pixl_Norm(x)
            # x = labels
            if is_smooth:
                epsilon = 0.1
                smooth = np.random.choice(2, self._category).astype(np.float32)
                c_num = np.sum(smooth)
                smooth[np.where(smooth==1.0)] = epsilon / c_num
                x = (1.0 - epsilon) * labels + smooth
            # x =  (1.0 - epsilon) * labels + epsilon / (1.0 * self._category)


            # mapping layers
            for layer_idx in range(mapping_layers):
                fmaps = self._channel * self.f_num
                out = lrelu(fully_connected(x, 512, use_bias=True, scope='dense%d' % layer_idx)) 
                out = lrelu(fully_connected(out, 256, use_bias=True, scope='dense1%d' % layer_idx)) 
                out = lrelu(fully_connected(out, fmaps, use_bias=True, scope='dense2%d' % layer_idx)) 
#                out = instance_norm(out, 'emb_y_proj_IN_{}'.format(layer_idx))
#            fmaps = self._channel * self.f_num
#            out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % 1))

            # if self.f_num == 1:
            #     out = out
            # else:
            #     out = tf.reshape(out, (-1, self._channel, self.f_num))
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Mapping_noaug():
    """LabelEmbed
    """
    def __init__(self,
                 channel=256,
                 activation=tf.nn.relu,
                 category=0,
                 f_num=1,
                 trainable=True,
                 name="Mapping"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.f_num = f_num

    def __call__(self, z_var, latent_size, batch_size, label_size=0, is_smooth=0, labels=None, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            mapping_layers = 4
            # x = z_var
            # latent_z = np.random.randn(batch_size, latent_size)
            # x = tf.cast(latent_z, tf.float32)
            if label_size:
                w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
                y = tf.matmul(labels, tf.cast(w, tf.float32))
                x = tf.concat([z_var, y], axis=1)

            # Normalize latents.
            # if norm_latent:
            # x = Pixl_Norm(x)
            x = labels
            if is_smooth:
                epsilon = 0.1
                smooth = np.random.choice(2, self._category).astype(np.float32)
                c_num = np.sum(smooth)
                smooth[np.where(smooth==1.0)] = epsilon / c_num
                x = (1.0 - epsilon) * labels + smooth
            # x =  (1.0 - epsilon) * labels + epsilon / (1.0 * self._category)


            # mapping layers
            for layer_idx in range(mapping_layers):
                fmaps = self._channel * self.f_num
                out = lrelu(fully_connected(x, fmaps, use_bias=True, scope='dense%d' % layer_idx))

            # if self.f_num == 1:
            #     out = out
            # else:
            #     out = tf.reshape(out, (-1, self._channel, self.f_num))
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_avg():
    """LabelEmbed
    """
    def __init__(self,
                 channel=16,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=38,
                 y_dim=2,
                 trainable=True,
                 f_num=4,
                 name="LabelEmbeds40_avg"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def __call__(self, latent_size, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            inputs = tf.transpose(labels, (1, 0, 2))  # [N,40,2]->[40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel*self.f_num), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]


            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32*4]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)
            # out4 = tf.matmul(out3, y_emb4)
            # out1 = tf.transpose(tf.reshape(out1, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))  #(f_num, N, 40, c)
            # out2 = tf.transpose(tf.reshape(out2, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))
            # out3 = tf.transpose(tf.reshape(out3, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))
            if self.f_num!=1:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel, self.f_num)), (3, 1, 0, 2))
            else:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel)), (1, 0, 2))  # (N, 40, c)
            # out2 = tf.reshape(out2, (-1, self.y_num, self._channel*4))
            # out3 = tf.reshape(out3, (-1, self.y_num, self._channel*2))
            # out4 = tf.reshape(out4, (-1, self.y_num, self._channel))

            # return out1, out2, out3, out4
            return out4
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_avg_z():
    """LabelEmbed
    """
    def __init__(self,
                 channel=16,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=38,
                 y_dim=2,
                 trainable=True,
                 f_num=4,
                 name="LabelEmbeds40_avg"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def __call__(self, z_var, latent_size, batch_size, labels=None, training = False, reuse = tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            inputs = tf.transpose(labels, (1, 0, 2))  # [N,40,2]->[40,N,2]
            # z_var = np.random.randn(self.y_num, batch_size, latent_size)  # [40,N,latent_size]
            # z_var = np.random.randn(batch_size, latent_size)  # [N,latent_size]
            # z_var = tf.tile(tf.expand_dims(z_var,0), (self.y_num,1,1)) # [40,N,latent_size]
            # z_var = tf.cast(z_var, tf.float32)
            #
            w = tf.get_variable('weight', shape=[self.y_num, self.y_dim, latent_size], dtype=tf.float32,
                                initializer=tf.initializers.random_normal())  # [40,2,latent_size]
            y = tf.matmul(inputs, tf.cast(w, tf.float32))  # [40,N,latent_size]
            inputs = tf.concat([z_var, y], axis=2) # [40,N,latent_size*2]
            inputs = Pixl_Norm(inputs)

            y_emb1 = tf.get_variable("emb1", (self.y_num, latent_size+z_var.shape[-1], self._channel*self.f_num), dtype=tf.float32,
            # y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel*self.f_num), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]


            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32*4]
            out1 = tf.nn.leaky_relu(out1)
            # out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tf.nn.leaky_relu(out2)
            # out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tf.nn.leaky_relu(out3)
            # out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tf.nn.leaky_relu(out4)
            # out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)
            # out4 = tf.matmul(out3, y_emb4)
            # out1 = tf.transpose(tf.reshape(out1, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))  #(f_num, N, 40, c)
            # out2 = tf.transpose(tf.reshape(out2, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))
            # out3 = tf.transpose(tf.reshape(out3, (self.y_num, -1, self._channel, self.f_num)), (3,1,0,2))
            if self.f_num!=1:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel, self.f_num)), (3, 1, 0, 2)) #(f_num, N, 40, c)
            else:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel)), (1, 0, 2))  # (N, 40, c)
            # out2 = tf.reshape(out2, (-1, self.y_num, self._channel*4))
            # out3 = tf.reshape(out3, (-1, self.y_num, self._channel*2))
            # out4 = tf.reshape(out4, (-1, self.y_num, self._channel))

            # return out1, out2, out3, out4
            return out4
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_avg_z2():
    """LabelEmbed
    """
    def __init__(self,
                 channel=16,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=38,
                 y_dim=2,
                 trainable=True,
                 f_num=4,
                 name="LabelEmbeds40_avg"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def __call__(self, latent_size, batch_size, labels=None, training = False, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputs = tf.transpose(labels, (1, 0, 2))  # [N,40,2]->[40,N,2]
            y = (inputs - 0.5)*2.0
            # z_var = np.random.randn(self.y_num, batch_size, latent_size)  # [40,N,latent_size]
            z_var = np.random.randn(batch_size, latent_size)  # [N,latent_size]
            z_var = tf.tile(tf.expand_dims(z_var,0), (self.y_num,1,1)) # [40,N,latent_size]
            z_var = tf.cast(z_var, tf.float32)

            # w = tf.get_variable('weight', shape=[self.y_num, self.y_dim, latent_size], dtype=tf.float32,
            #                     initializer=tf.initializers.random_normal())  # [40,2,latent_size]
            # y = tf.matmul(inputs, tf.cast(w, tf.float32))  # [40,N,latent_size]
            inputs = tf.concat([z_var, y], axis=2) # [40,N,latent_size*2]
            # inputs = Pixl_Norm(inputs)

            y_emb1 = tf.get_variable("emb1", (self.y_num, latent_size+self.y_dim, self._channel*self.f_num), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel*self.f_num, self._channel*self.f_num], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]


            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32*4]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)
            if self.f_num!=1:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel, self.f_num)), (3, 1, 0, 2))
            else:
                out4 = tf.transpose(tf.reshape(out4, (self.y_num, -1, self._channel)), (1, 0, 2))  # (N, 40, c)

            return out4
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_spatial():
    """LabelEmbed
    """
    def __init__(self,
                 channel=16,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=38,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_spatial"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            inputs = tf.transpose(labels, (1, 0, 2))  # [N,40,2]->[40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel*9), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel*9, self._channel*9], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel*9, self._channel*9], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel*9, self._channel*2], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]

            # bias1 = tf.get_variable("bias1", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias2 = tf.get_variable("bias2", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias3 = tf.get_variable("bias3", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())
            # bias4 = tf.get_variable("bias4", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())

            # out1 = tf.nn.relu(tf.matmul(inputs, y_emb1) + bias1)  # [40,N,32]
            # out2 = tf.nn.relu(tf.matmul(out1, y_emb2) + bias2)
            # out3 = tf.nn.relu(tf.matmul(out2, y_emb3) + bias3)
            # out4 = tf.nn.relu(tf.matmul(out3, y_emb4) + bias4)
            out1 = tf.matmul(inputs, y_emb1)  # [40,N,32*9]
            out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            out2 = tf.matmul(out1, y_emb2)
            out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            out3 = tf.matmul(out2, y_emb3)
            out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            out4 = tf.matmul(out3, y_emb4)
            out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)  # [40,N,32]
            # out4 = tf.reshape(out4, (-1, self.y_num, self._channel*2))
            # out4 = tf.matmul(out3, y_emb4)
            # out1 = tf.reshape(tf.transpose(out1,(1,0,2)), (4, -1, self.y_num, self._channel))
            # out2 = tf.reshape(tf.transpose(out2,(1,0,2)), (4, -1, self.y_num, self._channel))
            # out3 = tf.reshape(tf.transpose(out3,(1,0,2)), (4, -1, self.y_num, self._channel))
            # out4 = tf.reshape(tf.transpose(out4,(1,0,2)), (4, -1, self.y_num, self._channel))

            return out1, out2, out3, out4
            # return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds40_all32_bias():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 y_num=40,
                 y_dim=2,
                 trainable=True,
                 name="LabelEmbeds40_all32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.y_num = y_num
        self.y_dim = y_dim


    def __call__(self, batch_size, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            inputs = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            y_emb1 = tf.get_variable("emb1", (self.y_num, self.y_dim, self._channel), dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,2,32]
            y_emb2 = tf.get_variable("emb2", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb3 = tf.get_variable("emb3", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())  # [40,32,32]
            y_emb4 = tf.get_variable("emb4", [self.y_num, self._channel, self._channel], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())      #[40,32,32]
            # y_emb5 = tf.get_variable("emb5", [self.y_num, self._channel, self._channel], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,32,32]

            bias1 = tf.get_variable("bias1", (self.y_num, 1, self._channel), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            bias2 = tf.get_variable("bias2", (self.y_num, 1, self._channel), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            bias3 = tf.get_variable("bias3", (self.y_num, 1, self._channel), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            bias4 = tf.get_variable("bias4", (self.y_num, 1, self._channel), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            # bias5 = tf.get_variable("bias5", (self.y_num, 1, self._channel), dtype=tf.float32,
            #                         initializer=tf.zeros_initializer())

            out1 = tf.matmul(inputs, y_emb1) + bias1  # [40,N,32]
            out2 = tf.matmul(out1, y_emb2) + bias2
            out3 = tf.matmul(out2, y_emb3) + bias3
            out4 = tf.matmul(out3, y_emb4) + bias4
            # out5 = tf.matmul(out4, y_emb5) + bias5
            # out1 = tf.matmul(inputs, y_emb1)  # [40,N,32]
            # out1 = tcl.batch_norm(out1, activation_fn=tf.nn.relu, scale=True)
            # out1 = tf.nn.leaky_relu(out1)
            # out2 = tf.matmul(out1, y_emb2)
            # out2 = tcl.batch_norm(out2, activation_fn=tf.nn.relu, scale=True)
            # out2 = tf.nn.leaky_relu(out2)
            # out3 = tf.matmul(out2, y_emb3)
            # out3 = tcl.batch_norm(out3, activation_fn=tf.nn.relu, scale=True)
            # out3 = tf.nn.leaky_relu(out3)
            # out4 = tf.matmul(out3, y_emb4)
            # out4 = tcl.batch_norm(out4, activation_fn=tf.nn.relu, scale=True)
            # out4 = tf.nn.leaky_relu(out4)
            # out5 = tf.matmul(out4, y_emb5)
            # out5 = tcl.batch_norm(out5, activation_fn=tf.nn.relu, scale=True)
            # out5 = tf.nn.leaky_relu(out5)
            # out4 = tf.matmul(out3, y_emb4)
            out1 = tf.reshape(out1, (-1, self.y_num, self._channel))
            out2 = tf.reshape(out2, (-1, self.y_num, self._channel))
            out3 = tf.reshape(out3, (-1, self.y_num, self._channel))
            out4 = tf.reshape(out4, (-1, self.y_num, self._channel))
            # out5 = tf.reshape(out5, (-1, self.y_num, self._channel))

            return out1, out2, out3, out4
            # return out1
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbed256_cnn():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed64"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*1, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out5 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            # out6 = tcl.fully_connected(out, self._channel*16, activation_fn=tf.nn.relu,
            #                           normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out7 = tcl.fully_connected(out, self._channel*32, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            return out1, out2, out3, out4, out5
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbed64():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed64"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out5 = tcl.fully_connected(out4, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out6 = tcl.fully_connected(out5, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out7 = tcl.fully_connected(out6, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out8 = tcl.fully_connected(out7, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            return out1, out2, out3, out4, out5, out6, out7, out8
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbeds():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 z_dim=128,
                 f_num=2,
                 trainable=True,
                 name="LabelEmbeds"):
        self._channel = channel
        self._activation = activation
        # self._category = category
        self.f_num = f_num
        self.name = name
        self.z_dim = z_dim

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*self.f_num, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*self.f_num, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*self.f_num, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*self.f_num, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            if self.f_num == 1:
                out4 = out4
            else:
                out4 = tf.reshape(out4,(-1, self._channel, self.f_num))
            # return out1, out2, out3, out4
            return out4

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbeds_long():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 z_dim=128,
                 category=0,
                 trainable=True,
                 name="LabelEmbeds"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name
        self.z_dim = z_dim

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out5 = tcl.fully_connected(out, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out6 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out7 = tcl.fully_connected(out, self.z_dim, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            return out1, out2, out3, out4, out5, out6, out7
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbed32_cnn_long():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3, out4, out5, out6
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbed32_cnn():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*1, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*1, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*1, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*1, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            # return out1, out2, out3, out4
            #return out1, out2, out3
            return out4
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbed32_cnn_reverse():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel, activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3, out4
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbed32_cnn1():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed321"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out4 = tcl.fully_connected(out3, self._channel*8, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbed32_0():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3, out4
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class LabelEmbed32():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3, out4
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LabelEmbed32_1():
    """LabelEmbed
    """
    def __init__(self,
                 channel=32,
                 activation=tf.nn.relu,
                 category=0,
                 trainable=True,
                 name="LabelEmbed32_1"):
        self._channel = channel
        self._activation = activation
        self._category = category
        self.name = name

    def __call__(self, labels=None, training = True, reuse = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = labels
            
            out1 = tcl.fully_connected(out, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out2 = tcl.fully_connected(out1, self._channel*8, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out3 = tcl.fully_connected(out2, self._channel*4, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            out4 = tcl.fully_connected(out3, self._channel*2, activation_fn=tf.nn.relu, 
                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out5 = tcl.fully_connected(out4, self._channel*4, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
#            out6 = tcl.fully_connected(out5, self._channel*2, activation_fn=tf.nn.relu, 
#                                      normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training':training})
            
            return out1, out2, out3, out4
            #return out1, out2, out3
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
