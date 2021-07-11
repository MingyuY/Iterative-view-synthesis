#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:41:14 2020

@author: ymy
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:11:10 2019

@author: ubuntu
"""

import tensorflow.contrib.layers as tcl
from ops_coor import *
import tensorflow as tf 

class encoder_concat(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi


    def __call__(self, conv, batch_size, reuse=False, pg=4, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()
            # for i in range(self.y_num):
            #     label = labels[:, i, :]  # [N, 2]
            #     y = tf.reshape(label, (-1, 1, 1, self.y_dim))
            #     y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            #     conv = tf.concat([conv, y], 3)

            y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            conv = tf.concat([conv, y], 3)

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))

            for i in range(pg):
                res = conv

                res = lrelu(instance_norm(
                    conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                    scope='enc_IN_resblk_res_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                    scope='enc_IN_resblk_1_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                    scope='enc_IN_resblk_2_{}'.format(i)))

                conv = res + conv


            with tf.name_scope('conv_last') as scope:
                z_mu = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                z_mu = tf.reduce_mean(z_mu, axis=3, keep_dims=True)
                z_logvar = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma')
                z_logvar = tf.reduce_mean(z_logvar, axis=3, keep_dims=True)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style_share(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, name='styleganencoder_share'):
        self.name = name
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, reuse=False, pg=4, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            # conv = lrelu(instance_norm(
            #     conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg):
                res = conv

                res = instance_norm(conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                    scope='enc_IN_resblk_res_{}'.format(i))
                # res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                conv = instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                    scope='enc_IN_resblk_1_{}'.format(i))
                # conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                conv = instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                    scope='enc_IN_resblk_2_{}'.format(i))
                # conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d'%i)
                conv = lrelu(conv)

                conv = res + conv
            return conv

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style_musigma(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, name='styleganencoder_musigma'):
        self.name = name
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def __call__(self, conv, reuse=False, pg=4, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            with tf.name_scope('conv_last') as scope:
                z_mu = conv_2d(conv, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                # z_mu = conv_2d(conv, channels=4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu_1')
                z_mu = tf.reduce_mean(z_mu, axis=3, keep_dims=True)
                z_logvar = conv_2d(conv, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False,
                                   name='enc_sigma')
                # z_logvar = conv_2d(conv, channels=4, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma_1')
                z_logvar = tf.reduce_mean(z_logvar, axis=3, keep_dims=True)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style_recover(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, name='styleganencoder_recover'):
        self.name = name
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def __call__(self, conv, reuse=False, pg=4, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            with tf.name_scope('conv_last') as scope:
                out = conv_2d(conv, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                # z_mu = conv_2d(conv, channels=4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu_1')
                out = tf.reduce_mean(out, axis=3, keep_dims=True)
            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, name='styleganencoder'):
        self.name = name
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse:
            #     scope.reuse_variables()
            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg):
                res = conv

                # res = instance_norm(conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                #     scope='enc_IN_resblk_res_{}'.format(i))
                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                #     scope='enc_IN_resblk_1_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                #     scope='enc_IN_resblk_2_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d'%i)
                conv = lrelu(conv)

                conv = res + conv


            with tf.name_scope('conv_last') as scope:
                z_mu = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                # z_mu = conv_2d(conv, channels=4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu_1')
                z_mu = tf.reduce_mean(z_mu, axis=3, keep_dims=True)
                z_logvar = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma')
                # z_logvar = conv_2d(conv, channels=4, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma_1')
                z_logvar = tf.reduce_mean(z_logvar, axis=3, keep_dims=True)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style_sigma(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, is_cov=0):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi
        self.is_cov = is_cov

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg):
                res = conv

                # res = instance_norm(conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                #     scope='enc_IN_resblk_res_{}'.format(i))
                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                #     scope='enc_IN_resblk_1_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = instance_norm(
                    # conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                    # scope='enc_IN_resblk_2_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d'%i)
                conv = lrelu(conv)

                conv = res + conv


            with tf.name_scope('conv_last') as scope:
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                c_shape = conv.shape.as_list()
                z_mu, z_logvar = tf.nn.moments(conv, axes=[3], keep_dims=True)
                if self.is_cov:
                    reshape_z = tf.reshape(conv - z_mu, (batch_size, c_shape[1]*c_shape[2], c_shape[3]))
                    z_cov = tf.matmul(reshape_z, tf.transpose(reshape_z, (0,2,1)))/tf.cast(batch_size-1, tf.float32)
                    z_cov = tf.reshape(z_cov, (batch_size, c_shape[1], c_shape[2], c_shape[1]*c_shape[2]))
                    z_stats = tf.concat((z_mu, z_logvar, z_cov), -1)
                    conv_stats = lrelu(conv_2d(z_stats, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_mu1'))
                    conv_stats = lrelu(conv_2d(conv_stats, channels=self.channel*2, kernel=3, stride=1, pad=1, sn=False, name='enc_mu2'))
                    z_mu = conv_2d(conv_stats, channels=1, kernel=3, stride=1, pad=1, sn=False, name='enc_mu_out')
                    z_logvar = conv_2d(conv_stats, channels=1, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma_out')
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_style_SPADE_VI(object):
    #dis_as_v = []
    def __init__(self, in_dim=512, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi  
    def enc_adain_oir(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out 
    
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        if len(embs_in.shape.as_list())!=2:
            embs_in = tf.reshape(embs_in, (embs_in.shape.as_list()[0], -1))
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2]*2, input_shape[3]/2))  # (N, size**2, c)
        if input_shape[3]/2!=embs_in.shape.as_list()[-1]:
            embs_in = lrelu(fully_connected(embs_in, input_shape[3]/2, use_bias=True, scope=name))  
        embs = tf.reshape(embs_in, (-1, input_shape[3]/2, 1)) 
        att = tf.matmul(input_r, embs)  # (N, size**2, 1)
        att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 2)) 
        return att_weight
    
    def enc_adain(self, x, latent_in, training=False, name=None, mul=1):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()  
            conv = tanh(instance_norm(conv_2d(x, channels=25*2, kernel=3, stride=1, pad=1, sn=False, name='enc_conv_offset1'), scope='enc_IN_deformconv'))
            offset = mul*self.do_att(conv, latent_in, name='enc_c_conv', is_training=training) 
            out = flow_warping(x, offset)
            out = tf.concat([x, out], -1)
            return out, offset
        
    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, pa_again=False, t=False, is_mlp=True, alpha_trans=0.01,is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:  
            F = []
            input_shape = conv.shape.as_list()
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv0'), scope='IN_1')) 
            if pa_again:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel, kernel=4, stride=2, pad=1, sn=False,name='enc_conv5'), scope='IN_5'))
                
            conv = conv_2d(conv, channels=self.channel, kernel=3, stride=1, pad=1, sn=False,name='pono_enc_conv1') 
            conv, mean1, std1 = PONO(conv)
            conv = lrelu(conv)
            F.append(conv)
            
            conv = conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='pono_enc_conv2')
            conv, mean2, std2 = PONO(conv)
            conv = lrelu(conv)
            F.append(conv)
            
            conv = conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='pono_enc_conv3')
            conv, mean3, std3 = PONO(conv)
            conv = lrelu(conv)
            F.append(conv)
    
             
#            conv = conv_2d(conv, channels=self.channel, kernel=3, stride=1, pad=1, sn=False,name='enc_conv1') 
#            conv = lrelu(instance_norm(conv, scope='enc_IN_deformconv_0'))
#            
#            
#            conv = conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2')
#            conv = lrelu(instance_norm(conv, scope='enc_IN_deformconv_1'))
#              
#            conv = conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3')
#            conv = lrelu(instance_norm(conv, scope='enc_IN_deformconv_2'))
             
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))
 
            for i in range(pg):
                res = conv
 
                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                res = instance_norm(res, scope='enc_resblk_in%d'%i)
                res = lrelu(res)
 
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                conv = instance_norm(conv, scope='enc_conv_in%d'%i)
                conv = lrelu(conv) 
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                conv = instance_norm(conv, scope='enc_conv_in%d'%i)
                conv = lrelu(conv)

                conv = res + conv
            
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2')) 
#            conv = lrelu(instance_norm(
#                conv_2d(conv, channels=self.channel*2, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2')) 
             
            conv = tf.reshape(conv, [batch_size, -1])
#            # conv = tf.reduce_mean(conv, axis=(1, 2))
#
            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))
            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.in_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.in_dim, activation_fn=None)
                p_label = tcl.fully_connected(out, self.y_dim, activation_fn=None)
            return z_mu, z_logvar, p_label, F
#            return conv, [[mean3, std3],  [mean2,std2], [mean1, std1]]
#            with tf.name_scope('fc8') as scope:
##                z_mu_fc = tcl.fully_connected(out, self.z_dim, activation_fn=None) # n*1*1*256
##                z_logvar_fc = tcl.fully_connected(out, self.z_dim, activation_fn=None)
#                f_shape = conv.shape.as_list()
#                z_mu, z_logvar = tf.nn.moments(conv, axes=[1, 2], keep_dims=True) # n*1*1*c 
#                conv_c = conv-z_mu # n*4*4*128
#                conv_1 = tf.reshape(conv_c, (f_shape[0], f_shape[1]*f_shape[2], f_shape[3]))#n*16*128
#                var = tf.matmul(tf.transpose(conv_1,(0,2,1)), conv_1)/tf.cast(batch_size-1, tf.float32)#n*128*128
#                var = tf.nn.softmax(var, -1)
##                v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
#                var = tf.reshape(var, [batch_size, -1])
#                 
#                z_mu_sp, z_logvar_sp = tf.nn.moments(conv, axes=[3], keep_dims=True) # n*1*1*c 
#                conv_sp = conv-z_mu_sp # n*4*4*128
#                conv_1_sp = tf.reshape(conv_sp, (f_shape[0], f_shape[1]*f_shape[2], f_shape[3]))#n*16*128
#                var_sp = tf.matmul(conv_1_sp, tf.transpose(conv_1_sp,(0,2,1)))/tf.cast(batch_size-1, tf.float32)#n*128*128
#                var_sp = tf.nn.softmax(var_sp, -1)
##                v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
#                var_sp = tf.reshape(var_sp, [batch_size, -1])
#                z_mu_sp_1 = tf.reshape(z_mu_sp,[batch_size, -1]) 
#                z_logvar_sp_1 = tf.reshape(z_logvar_sp,[batch_size, -1]) 
#                
#                out_concat = tf.concat([tf.squeeze(z_mu), tf.squeeze(z_logvar), var, z_mu_sp_1, z_logvar_sp_1, var_sp], -1) 
#                out_concat = lrelu(fully_connected(out_concat, 1024, use_bias=True, sn=False, scope='o_2'))
#                z_mu_out = tcl.fully_connected(out_concat, self.z_dim, activation_fn=None) 
#                z_logvar_out = tcl.fully_connected(out_concat, self.z_dim, activation_fn=None) 
#            return z_mu_out , z_logvar_out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class LatentDiscriminator(object):
    def __init__(self, name='LatentDiscriminator', y_dim = 530):
        self.name = name
        self.y_dim = y_dim

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = tcl.fully_connected(inputs, 256, activation_fn=tf.nn.relu)
            out = tcl.fully_connected(out, self.y_dim, activation_fn=None)
            return out
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class encoder_style_PN(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[3], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            reshape_z = tf.reshape(conv - z_mu, (batch_size, c_shape[1]*c_shape[2], c_shape[3]))
            z_cov = tf.matmul(conv, tf.transpose(reshape_z, (0,2,1)))/tf.cast(batch_size-1, tf.float32)
            z_cov = tf.reshape(z_cov, (batch_size, c_shape[1], c_shape[2], c_shape[1]*c_shape[2]))
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out

    def __call__(self, conv, embs, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, is_mlp=True, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)
            #
            # # Update moving average of W.
            # with tf.variable_scope('zlAvg'):
            #     # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
            #     batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
            #     update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
            #     with tf.control_dependencies([update_op]):
            #         embs_l = tf.identity(embs_l)
            #     embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg):
                res = conv

                # res = instance_norm(conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                #     scope='enc_IN_resblk_res_{}'.format(i))
                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.enc_adain(res, embs, training=is_training, name='spade_res%d'%i)
                res = lrelu(res)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                #     scope='enc_IN_resblk_1_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                if is_mlp:
                    conv = self.enc_adain(conv, embs, training=is_training, name='spade1%d'%i)
                conv = lrelu(conv)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                #     scope='enc_IN_resblk_2_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                if is_mlp:
                    conv = self.enc_adain(conv, embs, training=is_training, name='spade2%d'%i)
                conv = lrelu(conv)

                conv = res + conv

            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2'))
            conv = tf.reshape(conv, [batch_size, -1])
            # conv = tf.reduce_mean(conv, axis=(1, 2))

            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class encoder_style_vector(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, is_mlp=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg):
                res = conv

                # res = instance_norm(conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                #     scope='enc_IN_resblk_res_{}'.format(i))
                res = conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                #     scope='enc_IN_resblk_1_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = instance_norm(
                #     conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                #     scope='enc_IN_resblk_2_{}'.format(i))
                conv = conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d'%i)
                conv = lrelu(conv)

                conv = res + conv

            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2'))
            conv = tf.reshape(conv, [batch_size, -1])
            # conv = tf.reduce_mean(conv, axis=(1, 2))

            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class encoder_cvaegan(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7, name='styleganencoder'):
        self.name = name
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi


    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=4, pa=False, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()
            y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            conv = tf.concat([conv, y], 3)

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))
            if pa:
                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv4'), scope='IN_4'))

            for i in range(pg):
                res = conv

                res = lrelu(instance_norm(
                    conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                    scope='enc_IN_resblk_res_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                    scope='enc_IN_resblk_1_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                    scope='enc_IN_resblk_2_{}'.format(i)))

                conv = res + conv

            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2'))
            conv = tf.reshape(conv, [batch_size, -1])
            # conv = tf.reduce_mean(conv, axis=(1, 2))

            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)

            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class generate_style_res_VI(object):
    #dis_as_v = []
    def     __init__(self, z_dim=256, y_num=40, y_dim=2, y_diff_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.y_diff_dim = y_diff_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 **(stage * 1)), 256)
 
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out
   
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        if len(embs_in.shape.as_list())!=2:
            embs_in = tf.reshape(embs_in, (embs_in.shape.as_list()[0], -1))
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2]*2, input_shape[3]/2))  # (N, size**2, c)
        if input_shape[3]/2!=embs_in.shape.as_list()[-1]:
            embs_in = lrelu(fully_connected(embs_in, input_shape[3]/2, use_bias=True, scope=name))  
        embs = tf.reshape(embs_in, (-1, input_shape[3]/2, 1)) 
        att = tf.matmul(input_r, embs)  # (N, size**2, 1)
        att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 2)) 
        return att_weight
    def KGconv(self, x, latent_in, training=False, name=None, mul=1):
        with tf.variable_scope(name or 'KGconv') as scope:
            if len(x)==2:
                x_dec = x[1]
                x = x[0]
                input_shape = x.shape.as_list()  
                conv = lrelu(instance_norm(conv_2d(tf.concat([x, x_dec], -1), channels=50, kernel=3, stride=1, pad=1, sn=False, \
                                                  name='conv_offset1'), scope='gen_IN_deformconv1'))
                conv = tanh(instance_norm(conv_2d(conv, channels=25*2, kernel=3, stride=1, pad=1, sn=False, \
                                                  name='conv_offset2'), scope='gen_IN_deformconv2'))
                conv = tanh(instance_norm(conv_2d(conv, channels=25*2, kernel=3, stride=1, pad=1, sn=False, \
                                                  name='conv_offset3'), scope='gen_IN_deformconv3'))
                
            else:
                x=x[0]
                input_shape = x.shape.as_list()  
                conv = tanh(instance_norm(conv_2d(x, channels=25*2, kernel=3, stride=1, pad=1, sn=False, name='conv_offset1'), scope='gen_IN_deformconv'))
            
            offset = mul*self.do_att(conv, latent_in, name='c_conv', is_training=training) 
            out = flow_warping(x, offset)
            return out, offset
    
    def enc_adain_oir(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out
    
    def Positional_cov(self, x1, x2, epsilon=1e-5):  
        f_shape1 = x1.shape.as_list()
        mean1, var1 = tf.nn.moments(x1, [3], keep_dims=True)   
        conv_c1 = x1-mean1 # n*4*4*128
        f_shape2 = x2.shape.as_list()
        mean2, var2 = tf.nn.moments(x2, [3], keep_dims=True)    
        conv_c2 = x2-mean2 # n*4*4*128
        conv_c1_r = tf.reshape(conv_c1, (f_shape1[0], f_shape1[1]*f_shape1[2], f_shape1[3]))#n*16*128
        conv_c2_r = tf.transpose(tf.reshape(conv_c2, (f_shape2[0], f_shape2[1]*f_shape2[2], f_shape2[3])),(0,2,1))#n*16*128
        cov = tf.matmul(conv_c1_r, conv_c2_r)/tf.cast(f_shape1[3]-1, tf.float32)#n*128*128
        cov = tf.reshape(cov, (f_shape1[0], f_shape1[1], f_shape1[2], f_shape1[1]*f_shape1[2]))
        cov = tf.nn.softmax(100*cov, -1)  
        return cov  
    def Channal_cov(self, x1, x2, epsilon=1e-5):  
        f_shape1 = x1.shape.as_list()
        mean1, var1 = tf.nn.moments(x1, [1,2], keep_dims=True)    
        conv_c1 = x1-mean1 # n*4*4*128
        f_shape2 = x1.shape.as_list()
        mean2, var2 = tf.nn.moments(x2, [1,2], keep_dims=True)    
        conv_c2 = x2-mean2 # n*4*4*128
        conv_c1_r = tf.transpose(tf.reshape(conv_c1, (f_shape1[0], f_shape1[1]*f_shape1[2], f_shape1[3])),(0,2,1))#n*16*128
        conv_c2_r = tf.reshape(conv_c2, (f_shape2[0], f_shape2[1]*f_shape2[2], f_shape2[3]))#n*16*128
        cov = tf.matmul(conv_c1_r, conv_c2_r)/tf.cast(f_shape2[1]*f_shape2[2]-1, tf.float32)#n*128*128
        cov = tf.reshape(cov, (f_shape1[0], f_shape1[3], f_shape1[3]))
        cov = tf.nn.softmax(100*cov, -1)  
        return cov
    
    def enc_adain(self, x, latent_in=None, training=False, name=None, mul=1):
        with tf.variable_scope(name or 'spade') as scope:
            x_kG, offset_KG = self.KGconv(x, latent_in, training=training, name=name+'KG', mul=mul)
            x_dec = x[1]
            x = x_kG
            x_shape = x.shape.as_list()
            input_shape = x.shape.as_list()  
            cov_p = self.Positional_cov(x_dec, x)
            cov_p_shape = cov_p.shape.as_list()
            cov_p_r = tf.reshape(cov_p, (cov_p_shape[0], cov_p_shape[-1], cov_p_shape[-1]))
            F_out1 = tf.matmul(cov_p_r, tf.reshape(x, (x_shape[0], x_shape[1]*x_shape[2], x_shape[3])))
            F_out1 = tf.reshape(F_out1, (x_shape[0], x_shape[1], x_shape[2], x_shape[3]))
            cov_c = self.Channal_cov(x_dec, F_out1)
            xxx = tf.reshape(tf.matmul(cov_c, tf.reshape(tf.transpose(F_out1, (0, 3, 1, 2)), (x_shape[0], x_shape[3], x_shape[1]*x_shape[2]))), (x_shape[0], x_shape[3], x_shape[1], x_shape[2]))
            F_out1 = tf.transpose(xxx, (0, 2, 3, 1))
            cov_p_r = tf.reshape(cov_p, (input_shape[0], input_shape[1], input_shape[2], input_shape[1]*input_shape[2]))
            
#            offset = mul*self.do_att(conv, latent_in, name='c_conv', is_training=training) 
#            out = flow_warping(x, offset)
            return F_out1, cov_p_r, offset_KG
        
    def enc_adain_coortooffset(self, x, latent_in, offset_0, training=False, name=None, mul=1):
        with tf.variable_scope(name or 'spade') as scope:
            cov_p = offset_0[0]
            x_dec = x[1]
            x = x[0]
            x_shape = x.shape.as_list()
            if len(offset_0)>1:
                offset_down = offset_0[1]
                offset_down_shape = offset_down.shape.as_list()
                offset_down = tf.image.resize_images(offset_down, (x_shape[1], x_shape[2]))*(x_shape[1]/offset_down_shape[1])
                x = flow_warping(x, offset_down)

            cov_p_shape = cov_p.shape.as_list()
            hh = x_shape[1]/cov_p_shape[1]
            ww = x_shape[1]/cov_p_shape[1]
            
            x_reshape = tf.reshape(x, (x_shape[0], x_shape[1]/hh, hh, x_shape[2]/ww, ww, x_shape[3]))
            x_reshape = tf.transpose(x_reshape, (0, 1, 3, 5, 2, 4))
            x_reshape = tf.reshape(x_reshape, (x_shape[0], cov_p_shape[1]*cov_p_shape[2], x_shape[3]*hh*ww))
            cov_p_r = tf.reshape(cov_p, (cov_p_shape[0], cov_p_shape[-1], cov_p_shape[-1]))
            F_out1 = tf.matmul(cov_p_r, x_reshape)
            F_out1 = tf.reshape(F_out1, (x_shape[0], cov_p_shape[1], cov_p_shape[2], x_shape[3], hh, ww))
            F_out1 = tf.transpose(F_out1, (0, 1, 4, 2, 5, 3))
            x = tf.reshape(F_out1, (x_shape[0], x_shape[1], x_shape[2], x_shape[3]))
            
            if len(offset_0)==3:
                offset_down = offset_0[2]
                offset_down_shape = offset_down.shape.as_list()
                offset_down = tf.image.resize_images(offset_down, (x_shape[1], x_shape[2]))*(x_shape[1]/offset_down_shape[1])
                x = flow_warping(x, offset_down)
            latent_in_ = tf.tile(tf.expand_dims(tf.expand_dims(latent_in, 1), 1), [1, x_shape[1], x_shape[2], 1]) 
            input_shape = x.shape.as_list()
            inputs = tf.concat([latent_in_, x, x_dec], -1)
            offset = mul*tanh(instance_norm(conv_2d(inputs, channels=2, kernel=3, stride=1, pad=1, sn=False, name='conv'), scope='in'))
            F_out = flow_warping(x, offset)
            return F_out, offset
        
#        
    def __call__(self, inputs, labels, label_diff, statis, batch_size=1, train_batch_size=1, pg=4, pa=False, pa_again=False, t=False, alpha_trans=0.0 , is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope: 
            
            embs = lrelu(fully_connected(labels, 64, use_bias=True, scope='dense3')) 
            embs = lrelu(fully_connected(embs, 256, use_bias=True, scope='dense4')) 
            
            
            shape = inputs.shape.as_list() 
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de_shape = de.shape.as_list()  
            label_l1 = tf.tile(tf.expand_dims(tf.expand_dims(labels, (1)), 2), [1, de_shape[1], de_shape[2], 1])
            de = tf.concat([de, label_l1], -1)
#            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)
            
            state3 = statis[0]
            state2 = statis[1]
            state1 = statis[2]
            
            diff_out_l2 = lrelu(fully_connected(label_diff, 128, use_bias=True, scope='diff_MLPdense1')) 
              
#            state3 = deconv(state3, channels=state3.shape.as_list()[-1], kernel=4, stride=2, sn=False, scope='gen_state_1')
#            state2 = deconv(state2, channels=state2.shape.as_list()[-1], kernel=4, stride=2, sn=False, scope='gen_state_2')
#            state1 = deconv(state1, channels=state1.shape.as_list()[-1], kernel=4, stride=2, sn=False, scope='gen_state_3')
             

            for i in range(pg): 
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                res = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
#                res = instance_norm(res, scope='res%d'%i)
                res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                de = self.layer_adain(de, embs, training=is_training, name='style1%d'%i) 
#                de = instance_norm(de, scope='style1%d'%i) 
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                de = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
#                de = instance_norm(de, scope='style2%d'%i) 
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

                de = res + de
#-----------------------------------------------decoder1: using corse target pose to compute covrivance matrix with encoder feature-------
            #To RGB
#            if pa:
#                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            
            diff_out_l2 = lrelu(fully_connected(diff_out_l2, 64, use_bias=True, scope='diff_MLPdense2')) 
            diff_out_l2_1 = tanh(fully_connected(diff_out_l2, 25, use_bias=True, scope='diff_MLPdense_3')) 
            de = layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1')
            
            newstate1, offset1_diff, offset1_diff_KG = self.enc_adain([state1, de], diff_out_l2_1, training=is_training, name='diff_gen_deformconv_1', mul=1)
            offset1_conf_diff = sigmoid(conv_2d(tf.concat([newstate1, offset1_diff, offset1_diff_KG], -1), channels=1, kernel=1, stride=1, pad=0, sn=False, name='diff_gen_conf1_conv'))
            newstate1 = relu(newstate1)
            de_diff = self.enc_adain_oir(de, newstate1, training=is_training, name='diff_gen_res1')
            de_diff = lrelu(de_diff)  
            de = de_diff
            
            diff_out_l2 = tanh(fully_connected(diff_out_l2, 64, use_bias=True, scope='diff_MLPdense3')) 
            diff_out_l2_2 = tanh(fully_connected(tf.concat([diff_out_l2, \
                                        tf.reduce_mean(tf.reduce_mean(offset1_diff, [1,2]), -1, keep_dims=True), \
                                        tf.reduce_mean(tf.reduce_mean(offset1_diff_KG, [1,2]), -1, keep_dims=True)], -1), 25, use_bias=True, scope='diff_MLPdense_2')) 
            de = layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2')
            
            newstate2, offset2_diff = self.enc_adain_coortooffset([state2, de], diff_out_l2_2, \
                                                        offset_0 = [offset1_diff, offset1_diff_KG], training=is_training, name='diff_gen_deformconv_2', mul=2)
            offset2_conf_diff = sigmoid(conv_2d(tf.concat([newstate2, offset2_diff], -1), channels=1, kernel=1, stride=1, pad=0, sn=False, name='diff_gen_conf2_conv'))
            newstate2 = relu(newstate2)
            de_diff = self.enc_adain_oir(de, newstate2, training=is_training, name='diff_gen_res2')
            de_diff = lrelu(de_diff) 
            
            de = de_diff 
            
            
            diff_out_l2 = tanh(fully_connected(diff_out_l2, 64, use_bias=True, scope='diff_MLPdense4')) 
            diff_out_l2_3 = tanh(fully_connected(tf.concat([diff_out_l2, tf.reduce_mean(offset2_diff, [1,2]),\
                                                            tf.reduce_mean(tf.reduce_mean(offset1_diff, [1,2]), -1, keep_dims=True)], -1),\
                                                 25, use_bias=True, scope='diff_MLPdense_1')) 
            de = layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l3'), scope='gen_LN_l3')
            
            newstate3, offset3_diff = self.enc_adain_coortooffset( [state3, de] , diff_out_l2_3, \
                                    offset_0 = [offset1_diff, offset1_diff_KG, offset2_diff], training=is_training, name='diff_gen_deformconv_3', mul=4)
            offset3_conf_diff = sigmoid(conv_2d(tf.concat([newstate3, offset3_diff], -1), channels=1, kernel=1, stride=1, pad=0, sn=False, name='diff_gen_conf3_conv'))
            newstate3 = relu(newstate3) 
            de_diff = self.enc_adain_oir(de, newstate3, training=is_training, name='diff_gen_res3')
            de_diff = lrelu(de_diff) 
            
            de = de_diff
            
            if pa_again:
                de = lrelu(layer_norm(deconv(de, channels=self.channel, kernel=4, stride=2, sn=False, scope='gen_conv_l4'), scope='gen_LN_l4'))
            
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l5'), scope='gen_LN_l5'))
            de = conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out')
            
#            conf_out = sigmoid(conv_2d(de, channels=1, kernel=1, stride=1, pad=0, sn=False, name='conf_out'))
            de  = tanh(de)
#-----------------------------------------------decoder2: using corse image(target pose) feature to output crose image-------
            
            return de, [offset3_diff, offset2_diff, offset1_diff, offset1_diff_KG,\
                        offset3_conf_diff, offset2_conf_diff, offset1_conf_diff]
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class generate_style_res(object):
    #dis_as_v = []
    def     __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 **(stage * 1)), 256)

    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         # out = style_mod(x, latent_in, use_bias=True)
    #         # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #         #                          normalizer_params={'scale': True, 'is_training': training})
    #         # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #         style_xs = []
    #         for i in range(self.y_num):
    #             style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
    #             style_xs.append(style_x)
    #         style_xs = tf.concat(style_xs, axis=3)
    #         # out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
    #                          # normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
    #         return style_xs
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out
#
#
    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out
        
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        if len(embs_in.shape.as_list())!=2:
            embs_in = tf.reshape(embs_in, (embs_in.shape.as_list()[0], -1))
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2]*18, input_shape[3]/18))  # (N, size**2, c)
        if input_shape[3]/18!=embs_in.shape.as_list()[-1]:
            embs_in = lrelu(fully_connected(embs_in, input_shape[3]/18, use_bias=True, scope=name))  
        embs = tf.reshape(embs_in, (-1, input_shape[3]/18, 1)) 
        att = tf.matmul(input_r, embs)  # (N, size**2, 1)
        att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 18)) 
        return att_weight
    
    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'spade') as scope:
            input_shape = x.shape.as_list()  
            conv = conv_2d(x, channels=25*18, kernel=3, stride=1, pad=1, sn=False, name='conv_offset1') 
            offset = self.do_att(conv, latent_in, name='c_conv', is_training=training) 
            out = deform_conv2d(x, offset, [3,3, input_shape[3],  input_shape[3]], activation = None, scope=None)
            return out
#    def do_mynorm(self, inputs, atts, name=None, is_training=False):  # only z, no label
#        input_shape = inputs.shape.as_list()
#        mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
##        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
#        x_ = (inputs - mu_x) / (sigma_x + 1e-6)
#
#        mu_y = atts[0]
#        sigma_y = atts[1]
#
#        adain = sigma_y * x_ + mu_y
#        # adain = (1+sigma_y) * x_ + mu_y
#        return adain
#    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
#
#        input_shape = inputs.shape.as_list()
#        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
#        embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
#        embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)
#
#        atts = []
#        for i in range(self.f_num):
#            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
#            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
#            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
#            att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))
#            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
#            atts.append(att_weight)
#        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
#        return atts
#    
#    def enc_adain(self, x, latent_in, training=False, name=None):
#        with tf.variable_scope(name or 'spade') as scope:
#            input_shape = x.shape.as_list()  
#            att = self.do_att(x, latent_in, name='att_1_{}'.format(x.shape[1]), is_training=training)
#            de = self.do_mynorm(x, att, is_training=training)
#        return de
    

    def __call__(self, inputs, en_embs, embs, pg=4, pa=False, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            input_shape = inputs.shape.as_list()

            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
            #             gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 16, 16, self.channel*4])
            # de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_4_conv')
            # de = Pixl_Norm(lrelu(de))

#            de = tf.get_variable('z_i_n', shape=(1, 4, 4, input_shape[-1]/16), initializer=tf.initializers.zeros(), trainable=True)
#            de = tf.tile(de, [shape[0], 1, 1, 1])
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)


            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)
            # emb_avg = tf.get_variable('latent_avg', shape=(1, self.y_num, embs.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            # input_const = tf.get_variable("const", shape=[1, 16, 16, self.channel*4], initializer=tf.initializers.ones())
            # de = tf.tile(tf.cast(input_const, dtype=tf.float32), [input_shape[0], 1, 1, 1])

            # if is_mlp:
            #     de1 = self.layer_adain(de, embs, training=is_training, name='style_in')
            #     de2 = self.enc_adain(de, en_embs, training=is_training, name='style_in')
            #     de = tf.concat((de1, de2), 3)
            # de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
            #                 normalizer_fn=None)

            for i in range(pg):

                # de = upscale(de, 2)
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                res1 = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                if is_mlp:
                    res2 = self.enc_adain(res, en_embs, training=is_training, name='res%d'%i)
                else:
                    res2 = instance_norm(res, scope='res%d'%i)
                res = tf.concat((res1, res2), 3)
                res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                de1 = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                if is_mlp:
                    de2 = self.enc_adain(de, en_embs, training=is_training, name='style1%d'%i)
                else:
                    de2 = instance_norm(de, scope='style1%d'%i)
                de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                de1 = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
                if is_mlp:
                    de2 = self.enc_adain(de, en_embs, training=is_training, name='style2%d'%i)
                else:
                    de2 = instance_norm(de, scope='style2%d'%i)
                de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # de = lrelu(de)

                de = res + de

            #To RGB
            de = lrelu(instance_norm(self.enc_adain(de, en_embs, training=is_training, name='gen_deformconv_0') , scope='gen_IN_deformconv_0'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out')

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class generate_cvaegan(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi


    def __call__(self, inputs, pg=4, t=False, pa=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            shape = inputs.shape.as_list()
            inputs = tf.concat((inputs, labels), 1)
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.channel * 4, k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
                        gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 16, 16, self.channel*4])
            # de = tf.reshape(de, [shape[0], 8, 8, self.channel*4])
            de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_4_conv')
            de = Pixl_Norm(lrelu(de))


            for i in range(pg):

                # de = upscale(de, 2)
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                res = instance_norm(res, scope='IN_resblk_res_{}'.format(i))
                res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                res = instance_norm(res, scope='IN_resblk_conv1_{}'.format(i))
                de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                res = instance_norm(res, scope='IN_resblk_conv2_{}'.format(i))
                de = lrelu(de)

                de = res + de

            #To RGB

            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = tanh(conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out'))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class stylegangenerate_style_res_spatial(object):
    # dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 ** (stage * 1)), 256)

    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         # out = style_mod(x, latent_in, use_bias=True)
    #         # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #         #                          normalizer_params={'scale': True, 'is_training': training})
    #         # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #         style_xs = []
    #         for i in range(self.y_num):
    #             style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
    #             style_xs.append(style_x)
    #         style_xs = tf.concat(style_xs, axis=3)
    #         out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
    #                          # normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
    #         return out
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out

    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out
    def __call__(self,  en_embs, embs, pg=5, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training=False,
                 reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            # input_shape = inputs.shape.as_list()
            # inputs = tf.concat((inputs, labels), 1)
            # de = tf.reshape(Pixl_Norm(inputs), [input_shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [input_shape[0], 4, 4, int(self.get_nf(1))])
            # de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(),
                                      trainable=False)
            # emb_avg = tf.get_variable('latent_avg', shape=(1, self.y_num, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            w = 7
            input_const = tf.get_variable("const", shape=[1, w, w, self.get_nf(1)], initializer=tf.initializers.zeros())
            de = tf.tile(tf.cast(input_const, dtype=tf.float32), [en_embs.shape.as_list()[0], 1, 1, 1])
            if is_mlp:
                de1 = self.layer_adain(de, embs, training=is_training, name='style_in')
                de2 = self.enc_adain(de, en_embs, training=is_training, name='style_in')
                de = tf.concat((de1, de2), 3)
            de = tcl.conv2d(de, num_outputs=self.get_nf(1), kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    # To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                en_embs = deconv(en_embs, channels=1, kernel=4, stride=2, scope='up_enstyle%d'%i)
                res = de
                res = conv2d(res, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale,
                             name='gen_resblk_res_{}'.format(res.shape[1]))
                if is_mlp:
                    res1 = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                    res2 = self.enc_adain(res, en_embs, training=is_training, name='res%d'%i)
                    res = tf.concat((res1, res2), 3)
                res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale,
                            name='gen_resblk_conv1_{}'.format(de.shape[1]))
                if is_mlp:
                    de1 = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                    de2 = self.enc_adain(de, en_embs, training=is_training, name='style1%d'%i)
                    de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale,
                            name='gen_resblk_conv2_{}'.format(de.shape[1]))
                if is_mlp:
                    de1 = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
                    de2 = self.enc_adain(de, en_embs, training=is_training, name='style2%d'%i)
                    de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

                de = res + de

            # To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1,
                        name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class encoder_concat_vector(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi


    def __call__(self, conv, batch_size, reuse=False, pg=4, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()
            # for i in range(self.y_num):
            #     label = labels[:, i, :]  # [N, 2]
            #     y = tf.reshape(label, (-1, 1, 1, 2))
            #     y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            #     conv = tf.concat([conv, y], 3)

            y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            conv = tf.concat([conv, y], 3)

            # fromRGB
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel, kernel=7, stride=1, pad=3, sn=False,name='enc_conv1'), scope='IN_1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*2, kernel=4, stride=2, pad=1, sn=False,name='enc_conv2'), scope='IN_2'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv3'), scope='IN_3'))

            for i in range(pg):
                res = conv

                res = lrelu(instance_norm(
                    conv_2d(res, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_res_{}'.format(i)),
                    scope='enc_IN_resblk_res_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv1_{}'.format(i)),
                    scope='enc_IN_resblk_1_{}'.format(i)))

                conv = lrelu(instance_norm(
                    conv_2d(conv, channels=self.channel*4, kernel=3, stride=1, pad=1, sn=False, name='enc_resblk_conv2_{}'.format(i)),
                    scope='enc_IN_resblk_2_{}'.format(i)))

                conv = res + conv

            # conv = lrelu(
            #     conv2d(conv, output_dim=self.channel*4, k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale,
            #            sn=False, name='enc_n_conv_1_{}'.format(conv.shape[1])))
            # conv = lrelu(
            #     conv2d(conv, output_dim=self.channel*4, k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale,
            #            sn=False, padding='VALID', name='enc_n_conv_2_{}'.format(conv.shape[1])))
            # out = tf.reduce_mean(conv, axis=(1, 2))

            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=4, stride=2, pad=1, sn=False,name='enc_conv_l1'), scope='IN_conv_l1'))
            conv = lrelu(instance_norm(
                conv_2d(conv, channels=self.channel*4, kernel=3, stride=2, pad=1, sn=False,name='enc_conv_l2'), scope='IN_conv_l2'))
            conv = tf.reshape(conv, [batch_size, -1])

            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))
            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class generate_style_vector(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 **(stage * 1)), 256)

    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         # out = style_mod(x, latent_in, use_bias=True)
    #         # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #         #                          normalizer_params={'scale': True, 'is_training': training})
    #         # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #         style_xs = []
    #         for i in range(self.y_num):
    #             style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
    #             style_xs.append(style_x)
    #         style_xs = tf.concat(style_xs, axis=3)
    #         out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
    #                          # normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
    #         return out
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out

    # def enc_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
    #         x_ = (x - mu_x) / (sigma_x + 1e-6)
    #
    #         conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
    #         mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
    #         sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')
    #
    #         out = sigma_y * x_ + mu_y
    #         return out

    def __call__(self, inputs, embs, pg=4, t=False, pa=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            # de = inputs
            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
            #             gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 16, 16, self.channel*4])
            # de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_4_conv')
            # de = Pixl_Norm(lrelu(de))

            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            for i in range(pg):

                # de = upscale(de, 2)
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                # res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                if is_mlp:
                    de = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                # de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                if is_mlp:
                    de = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
                # de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = lrelu(de)

                de = res + de

            #To RGB

            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = tanh(conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out'))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)



class generate_style_SPADE(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 **(stage * 1)), 256)

    def enc_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
            x_ = (x - mu_x) / (sigma_x + 1e-6)

            conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
            mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
            sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')

            out = sigma_y * x_ + mu_y
            return out

    def __call__(self, inputs, embs, pg=4, t=False, pa=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            # de = inputs
            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
            #             gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 16, 16, self.channel*4])
            # de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_4_conv')
            # de = Pixl_Norm(lrelu(de))

            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)

            for i in range(pg):

                # de = upscale(de, 2)
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                if is_mlp:
                    res = self.enc_adain(res, embs, training=is_training, name='spade%d'%i)
                # res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                if is_mlp:
                    de = self.enc_adain(de, embs, training=is_training, name='spade1%d'%i)
                # de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                if is_mlp:
                    de = self.enc_adain(de, embs, training=is_training, name='spade2%d'%i)
                # de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = lrelu(de)

                de = res + de

            #To RGB

            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out')

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class generate_style_res_vector(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(1024 / (2 **(stage * 1)), 512)
        return min(512 / (2 **(stage * 1)), 256)

    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         # out = style_mod(x, latent_in, use_bias=True)
    #         # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #         #                          normalizer_params={'scale': True, 'is_training': training})
    #         # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #         style_xs = []
    #         for i in range(self.y_num):
    #             style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
    #             style_xs.append(style_x)
    #         style_xs = tf.concat(style_xs, axis=3)
    #         out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
    #                          # normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
    #         return out
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out

    # def enc_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         mu_x, sigma_x = tf.nn.moments(x, axes=[0,1,2], keep_dims=True)
    #         x_ = (x - mu_x) / (sigma_x + 1e-6)
    #
    #         conv = conv_2d(latent_in, channels=self.channel, kernel=3, stride=1, pad=1, sn=False, name='conv')
    #         mu_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='mu')
    #         sigma_y = conv_2d(conv, channels=input_shape[3], kernel=3, stride=1, pad=1, sn=False, name='sigma')
    #
    #         out = sigma_y * x_ + mu_y
    #         return out

    def __call__(self, inputs, en_embs, embs, pg=4, pa=False, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            # if reuse == True:
            #     scope.reuse_variables()

            input_shape = inputs.shape.as_list()

            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
            #             gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 16, 16, self.channel*4])
            # de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_4_conv')
            # de = Pixl_Norm(lrelu(de))

            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 4, 4, -1])
            de = lrelu(conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_n_1_conv'))
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_2_conv')
            de = lrelu(de)
            de = deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_n_3_conv')
            de = lrelu(de)

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            for i in range(pg):

                # de = upscale(de, 2)
                res = de
                res = conv_2d(res, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_res_{}'.format(i))
                if is_mlp:
                    res1 = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                    # res2 = self.enc_adain(res, en_embs, training=is_training, name='res%d'%i)
                    res2 = self.layer_adain(res, en_embs, training=is_training, name='enc_res%d'%i)
                    res = tf.concat((res1, res2), 3)
                res = tcl.conv2d(res, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # res = lrelu(res)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv1_{}'.format(i))
                if is_mlp:
                    de1 = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                    # de2 = self.enc_adain(de, en_embs, training=is_training, name='style1%d'%i)
                    de2 = self.layer_adain(de, en_embs, training=is_training, name='enc_style1%d'%i)
                    de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # de = lrelu(de)

                de = conv_2d(de, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='gen_resblk_conv2_{}'.format(i))
                if is_mlp:
                    de1 = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
                    # de2 = self.enc_adain(de, en_embs, training=is_training, name='style2%d'%i)
                    de2 = self.layer_adain(de, en_embs, training=is_training, name='enc_style2%d'%i)
                    de = tf.concat((de1, de2), 3)
                de = tcl.conv2d(de, num_outputs=self.channel * 4, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # de = lrelu(de)

                de = res + de

            #To RGB

            de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l1'), scope='gen_LN_l1'))
            if pa:
                de = lrelu(layer_norm(deconv(de, channels=self.channel * 4, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l2'), scope='gen_LN_l2'))
            # if pa:
            #     de = lrelu(layer_norm(deconv(de, channels=self.channel * 2, kernel=4, stride=2, sn=False, scope='gen_conv_l21'), scope='gen_LN_l21'))
            de = lrelu(layer_norm(conv_2d(de, channels=self.channel * 1, kernel=7, stride=1, pad=3, sn=False, name='gen_conv_l3'), scope='gen_LN_l3'))
            de = conv_2d(de, channels=3, kernel=1, stride=1, pad=0, sn=False, name='gen_out')

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class discriminator_img(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32):
        self.name = 'discriminator'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel

    def __call__(self, inputs, batch_size, labels=None, pg=4, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            conv = tf.nn.leaky_relu(conv_2d(inputs, channels=self.channel * 1, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(1)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 2, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(2)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 4, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(3)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 8, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(4)))

            # conv = conv_2d(conv, channels=self.channel * 8, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(4))

            conv = tf.reshape(conv, [batch_size, -1])

            # for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')

            hidden = conv
            emb_y = fully_connected(labels, conv.shape[-1], use_bias=False, sn=True, scope='y_proj')
            proj = tf.reduce_sum(emb_y * hidden, axis=1, keep_dims=True)

            # if self.y_num > 1:
            #     cls = []
            #     for i in range(self.y_num):
            #         # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
            #         clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #         cls.append(clsi)
            #     y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            # else:
            #     y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')

            # return output, y_cls
            return output + proj
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class classifier_img(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, f_num=2, channel=32):
        self.name = 'classifier'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel

    def __call__(self, inputs, batch_size, labels=None, pg=4, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            conv = tf.nn.leaky_relu(conv_2d(inputs, channels=self.channel * 1, kernel=3, stride=1, pad=1, sn=False, name='cls_conv{}'.format(1)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=False, name='cls_conv{}'.format(2)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=False, name='cls_conv{}'.format(3)))
            conv = tf.nn.leaky_relu(conv_2d(conv, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=False, name='cls_conv{}'.format(4)))

            # conv = conv_2d(conv, channels=self.channel * 8, kernel=4, stride=2, pad=1, sn=False, name='dis_conv{}'.format(4))

            conv = tf.reshape(conv, [batch_size, -1])

            if self.y_num > 1:
                cls = []
                for i in range(self.y_num):
                    # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
                    clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
                    cls.append(clsi)
                y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            else:
                y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')

            return y_cls

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
