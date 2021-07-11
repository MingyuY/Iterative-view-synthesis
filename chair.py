#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:38:36 2020

@author: ymy
"""

import os, sys
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim 
slim = contrib_slim
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))

# from models.discriminator_my import *
# from models.generator_my import *
# from models.encoder_my import *
from models.mynets_celeb import *
#from models.aguit_nets import *
from models.cascade_multiPIE_cVAEGANskip_CTM_flowloss_cascadeflow_GWimg_cascadeW import *
from datasets.datas import *
from models.label_embed import *
from datasets.get_data import *
from tensorflow.contrib.image import dense_image_warp
#hinge loss
#def Y_to_matrix(y):
#    map_dic = {}
import math
from tqdm import tqdm 

def flow_warping(x, flow):
    x_shape = x.shape.as_list()
    flow_shape = flow.shape.as_list()
    assert x_shape[:-1]==flow_shape[:-1]
    assert flow_shape[-1]==2
    return dense_image_warp(x, flow)

def soft_flow_warping_all(x, cov_p):
    x_shape = x.shape.as_list()
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
    F_out1 = tf.reshape(F_out1, (x_shape[0], x_shape[1], x_shape[2], x_shape[3])) 
    return F_out1 

def Channal_cov(x1, x2, epsilon=1e-5):  
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
    cov = tf.nn.softmax(cov, -1)  
    return cov  

def soft_flow_warping(x, flow):
    cov_p = flow
    x_shape = x.shape.as_list()
    cov_p_shape = cov_p.shape.as_list()
    cov_p_r = tf.reshape(cov_p, (cov_p_shape[0], cov_p_shape[-1], cov_p_shape[-1]))
    F_out1 = tf.matmul(cov_p_r, tf.reshape(x, (x_shape[0], x_shape[1]*x_shape[2], x_shape[3])))
    F_out1 = tf.reshape(F_out1, (x_shape[0], x_shape[1], x_shape[2], x_shape[3]))
    
    cov_c = Channal_cov(x, F_out1)
    
    yyy = tf.reshape(tf.transpose(F_out1, (0, 3, 1, 2)), (x_shape[0], x_shape[3], x_shape[1]*x_shape[2]))
    xxx = tf.reshape(tf.matmul(cov_c, yyy), (x_shape[0], x_shape[3], x_shape[1], x_shape[2]))
    F_out = tf.transpose(xxx, (0, 2, 3, 1))
    return F_out1
#hinge loss
#def Y_to_matrix(y):
#    map_dic = {} 
def Y_to_matrix(t, y_dim):#a e d
    p_shape = t.shape.as_list() 
    t = tf.cast(t, dtype=tf.float32) 
    a = t*(180.0/y_dim)*2*math.pi/360 
    matrix_z = tf.reshape([[tf.cos(a), -tf.sin(a), tf.constant(0.0,shape=(p_shape[0],))],
                            [tf.sin(a), tf.cos(a), tf.constant(0.0,shape=(p_shape[0],))],
                            [tf.constant(0.0,shape=(p_shape[0],)), tf.constant(0.0,shape=(p_shape[0],)), tf.constant(1.0, shape=(p_shape[0],))]],\
                            (p_shape[0],3,3))
    R = tf.reshape( matrix_z , (p_shape[0], -1)) 
    return R 

def loss_hinge_dis(dis_fake, dis_real):
#    loss = 1*tf.reduce_mean(tf.nn.relu(1. - dis_real))
    loss = tf.reduce_mean(-dis_real)
    loss += tf.reduce_mean(dis_fake)
#    loss += tf.reduce_mean(tf.nn.relu(1. + dis_fake))/3.0 
    return loss

def loss_hinge_gen(dis_fake):
    loss = -tf.reduce_mean(dis_fake)
    # loss = -tf.reduce_mean(tf.nn.softplus(dis_fake))
    return loss

def sample_z(m, n):
    # return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(0, 1, size=[m, n])

def sample_z1(m, n, c):
    # return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(0, 1, size=[m, n, n, c])

def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))

def kl_loss(avg, log_var):
    with tf.name_scope('KLLoss'):
        # return tf.reduce_mean(0.5 * tf.reduce_sum(-1.0 - log_var + tf.square(avg) + tf.exp(log_var), axis=-1))
        return tf.reduce_mean(0.5 * tf.reduce_mean(tf.exp(log_var) + avg ** 2 - 1. - log_var, 1))  #log_var=log(sigma^2)

def loss_cls(y_cls_logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_cls_logit, labels=label))
    return loss

class SNGAN():
    def __init__(self, encoder, generator, discriminator, latent_discriminator, batch_size, train_batch_size, log_dir='logs/imagenet',
                 model_dir='models/imagenet/', learn_rate_init=2e-4):
        self.log_vars = []
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.size = 128
        self.batch_size = batch_size 
        self.train_batch_size = train_batch_size
        self.train_file = ['/datasets/chair128_train_paired.tfrecords'] 
        self.test_file = ['/datasets/chair128_test_paired.tfrecords'] 
        self.imga, self.labela, self.pa, self.ta, self.imgb, self.labelb, self.pb, self.tb = read_and_decode_chair_paired_new(self.train_file, self.size)
        self.img_batcha, self.label_batcha, self.img_batchb, self.label_batchb = tf.train.shuffle_batch([self.imga, self.labela, self.imgb, self.labelb, ],
                                                                  batch_size=self.batch_size,
                                                                  capacity=1000, min_after_dequeue=600)
        self.imga_test, self.labela_test, _, _, self.imgb_test, self.labelb_test, _, _ = read_and_decode_chair_paired_new(self.test_file, self.size)
        self.img_batcha_test, self.label_batcha_test, self.img_batchb_test, self.label_batchb_test = tf.train.batch([self.imga, self.labela, self.imgb, self.labelb, ],
                                                                  batch_size=self.batch_size,
                                                                  capacity=1000)
        self.is_wganGP = True
        self.gp_lambda = 10
        self.learn_rate_init = learn_rate_init
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator 
        self.latent_discriminator = latent_discriminator
        self.in_dim = 512
        self.y_num = 1
        self.y_dim = 62
        self.y_diff_dim = 62
        self.channel = 3
        self.latent_size = 128
        self.w = 5
        self.c = 1
        self.pa = True
        self.pg = 4
        self.z_in = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_dim], name='z_in')
        self.Xa = tf.placeholder(tf.float32, shape=[self.batch_size, self.size, self.size, self.channel], name='Xa')
        self.Xb = tf.placeholder(tf.float32, shape=[self.batch_size, self.size, self.size, self.channel], name='Xb')
        self.Ya = tf.placeholder(tf.int32, shape=[self.batch_size], name='Ya')
        self.Yb = tf.placeholder(tf.int32, shape=[self.batch_size], name='Yb')
        self.training = tf.placeholder(tf.bool,  name='training')
        self.Ya_matrix = tf.one_hot(self.Ya, self.y_dim)
        self.Yb_matrix = tf.one_hot(self.Yb, self.y_dim)
        self.Y_diff_onehot = self.Yb_matrix - self.Ya_matrix
        self.Y_diff_rec_onehot = self.Ya_matrix - self.Yb_matrix 
        self.Y_diff_self_onehot = self.Ya_matrix - self.Ya_matrix
        
        self.z_mu, self.z_log_var, self.p_label_a, self.statis_a = self.encoder(self.Xa, batch_size=self.batch_size, pg=3, pa=self.pa , is_mlp=False, is_training = self.training)
        self.z_f = sample_normal(self.z_mu, self.z_log_var)
        self.c_enc_p = self.latent_discriminator(self.z_f)
        
        self.x_b_fake, self.x_b_fake_diff_list = self.generator(self.z_f, labels=self.Yb_matrix, label_diff=self.Y_diff_onehot, statis = self.statis_a, batch_size=self.batch_size, train_batch_size=self.train_batch_size, pa=self.pa, pg=3, is_mlp=False, is_training=self.training) 
        self.x_b_fake_diff_offset3, self.x_b_fake_diff_offset2, self.x_b_fake_diff_offset1, self.x_b_fake_diff_offset1_KG, self.x_b_fake_diff_conf3, self.x_b_fake_diff_conf2, self.x_b_fake_diff_conf1 = self.x_b_fake_diff_list
 
        self.x_aa_fake, self.x_aa_fake_diff_list = self.generator(self.z_f, labels=self.Ya_matrix, label_diff=self.Y_diff_self_onehot, statis = self.statis_a, batch_size=self.batch_size, train_batch_size=self.train_batch_size, pa=self.pa, pg=3, is_mlp=False, is_training=self.training, reuse=True)
        self.x_aa_fake_diff_offset3, self.x_aa_fake_diff_offset2, self.x_aa_fake_diff_offset1, self.x_aa_fake_diff_offset1_KG, self.x_aa_fake_diff_conf3, self.x_aa_fake_diff_conf2, self.x_aa_fake_diff_conf1 = self.x_aa_fake_diff_list
 
        self.z_mu_rec, self.z_log_var_rec, self.p_label_b, self.statis_b = self.encoder(self.x_b_fake, reuse=True, batch_size=self.batch_size, pg=3, pa=self.pa , is_mlp=False, is_training = self.training)
        self.z_f_rec = sample_normal(self.z_mu_rec, self.z_log_var_rec)
        self.c_enc_p_rec = self.latent_discriminator(self.z_f_rec, reuse=True)
        
        self.x_a_fake , self.x_a_fake_diff_list = self.generator(self.z_f_rec, labels=self.Ya_matrix, label_diff=self.Y_diff_rec_onehot, statis = self.statis_b, batch_size=self.batch_size, train_batch_size=self.train_batch_size, pa=self.pa, pg=3, is_mlp=False, is_training=self.training, reuse=True)
        self.x_a_fake_diff_offset3, self.x_a_fake_diff_offset2, self.x_a_fake_diff_offset1, self.x_a_fake_diff_offset1_KG, self.x_a_fake_diff_conf3, self.x_a_fake_diff_conf2, self.x_a_fake_diff_conf1 = self.x_a_fake_diff_list

#------flow warp image loss-----------------------------------------
        size_128 = self.size/1 
        self.x_b_fake_diff_all_offset3 = tf.image.resize_images(self.x_b_fake_diff_offset2, (size_128, size_128))*2 + self.x_b_fake_diff_offset3
        self.x_a_fake_diff_all_offset3 = tf.image.resize_images(self.x_a_fake_diff_offset2, (size_128, size_128))*2 + self.x_a_fake_diff_offset3
        self.x_aa_fake_diff_all_offset3 = tf.image.resize_images(self.x_aa_fake_diff_offset2, (size_128, size_128))*2 + self.x_aa_fake_diff_offset3
        all_diff_fake_b_128 = flow_warping(soft_flow_warping_all(flow_warping(self.Xa, \
                            tf.image.resize_images(self.x_b_fake_diff_offset1_KG, (size_128, size_128))*4), self.x_b_fake_diff_offset1), self.x_b_fake_diff_all_offset3)
        all_diff_fake_a_128 = flow_warping(soft_flow_warping_all(flow_warping(self.x_b_fake, \
                            tf.image.resize_images(self.x_a_fake_diff_offset1_KG, (size_128, size_128))*4), self.x_a_fake_diff_offset1), self.x_a_fake_diff_all_offset3)
        all_diff_fake_aa_128 = flow_warping(soft_flow_warping_all(flow_warping(self.Xa, \
                            tf.image.resize_images(self.x_aa_fake_diff_offset1_KG, (size_128, size_128))*4), self.x_aa_fake_diff_offset1), self.x_aa_fake_diff_all_offset3)
        
        self.all_diffor_fake_b_128 = self.x_b_fake_diff_conf3*all_diff_fake_b_128 + (1-self.x_b_fake_diff_conf3)*tf.image.resize_images(self.x_b_fake, (size_128, size_128))
        self.all_diffor_fake_a_128 = self.x_a_fake_diff_conf3*all_diff_fake_a_128 + (1-self.x_a_fake_diff_conf3)*tf.image.resize_images(self.x_a_fake, (size_128, size_128))
        self.all_diffor_fake_aa_128 = self.x_aa_fake_diff_conf3*all_diff_fake_aa_128 + (1-self.x_aa_fake_diff_conf3)*tf.image.resize_images(self.x_aa_fake, (size_128, size_128))
        
        self.loss_rec_warp3 = tf.reduce_mean(tf.square(self.Xb - self.all_diffor_fake_b_128)) +\
                             tf.reduce_mean(tf.square(self.Xa - self.all_diffor_fake_a_128)) +\
                             tf.reduce_mean(tf.square(self.Xa - self.all_diffor_fake_aa_128))              
         
        size_64 = self.size/2
        self.x_b_fake_diff_all_offset2 = self.x_b_fake_diff_offset2
        self.x_a_fake_diff_all_offset2 = self.x_a_fake_diff_offset2
        self.x_aa_fake_diff_all_offset2 = self.x_aa_fake_diff_offset2
        all_diff_fake_b_64 = flow_warping(soft_flow_warping_all(flow_warping(tf.image.resize_images(self.Xa, (size_64, size_64)), \
                                        tf.image.resize_images(self.x_b_fake_diff_offset1_KG, (size_64, size_64))*2), self.x_b_fake_diff_offset1),\
                                          self.x_b_fake_diff_all_offset2)
        all_diff_fake_a_64 = flow_warping(soft_flow_warping_all(flow_warping(tf.image.resize_images(self.x_b_fake, (size_64, size_64)), \
                                        tf.image.resize_images(self.x_a_fake_diff_offset1_KG, (size_64, size_64))*2), self.x_a_fake_diff_offset1), \
                                          self.x_a_fake_diff_all_offset2)
        all_diff_fake_aa_64 = flow_warping(soft_flow_warping_all(flow_warping(tf.image.resize_images(self.Xa, (size_64, size_64)),\
                                        tf.image.resize_images(self.x_aa_fake_diff_offset1_KG, (size_64, size_64))*2), self.x_aa_fake_diff_offset1), \
                                           self.x_aa_fake_diff_all_offset2)
        
        self.all_diffor_fake_b_64 = self.x_b_fake_diff_conf2*all_diff_fake_b_64 + (1-self.x_b_fake_diff_conf2)*tf.image.resize_images(self.x_b_fake, (size_64, size_64))
        self.all_diffor_fake_a_64 = self.x_a_fake_diff_conf2*all_diff_fake_a_64 + (1-self.x_a_fake_diff_conf2)*tf.image.resize_images(self.x_a_fake, (size_64, size_64))
        self.all_diffor_fake_aa_64 = self.x_aa_fake_diff_conf2*all_diff_fake_aa_64 + (1-self.x_aa_fake_diff_conf2)*tf.image.resize_images(self.x_aa_fake, (size_64, size_64))
        
        self.loss_rec_warp2 = tf.reduce_mean(tf.square(tf.image.resize_images(self.Xb, (size_64, size_64)) - self.all_diffor_fake_b_64)) +\
                             tf.reduce_mean(tf.square(tf.image.resize_images(self.Xa, (size_64, size_64)) - self.all_diffor_fake_a_64)) +\
                             tf.reduce_mean(tf.square(tf.image.resize_images(self.Xa, (size_64, size_64)) - self.all_diffor_fake_aa_64))   
        
        size_32 = self.size/4 
        all_diff_fake_b_32 = soft_flow_warping_all(flow_warping(tf.image.resize_images(self.Xa, (size_32, size_32)), \
                              self.x_b_fake_diff_offset1_KG ), self.x_b_fake_diff_offset1)
        all_diff_fake_a_32 = soft_flow_warping_all(flow_warping(tf.image.resize_images(self.x_b_fake, (size_32, size_32)), \
                              self.x_a_fake_diff_offset1_KG ), self.x_a_fake_diff_offset1)
        all_diff_fake_aa_32 = soft_flow_warping_all(flow_warping(tf.image.resize_images(self.Xa, (size_32, size_32)), \
                              self.x_aa_fake_diff_offset1_KG ), self.x_aa_fake_diff_offset1)
        
        self.all_diffor_fake_b_32 = self.x_b_fake_diff_conf1*all_diff_fake_b_32 + (1-self.x_b_fake_diff_conf1)*tf.image.resize_images(self.x_b_fake, (size_32, size_32))
        self.all_diffor_fake_a_32 = self.x_a_fake_diff_conf1*all_diff_fake_a_32 + (1-self.x_a_fake_diff_conf1)*tf.image.resize_images(self.x_a_fake, (size_32, size_32))
        self.all_diffor_fake_aa_32 = self.x_aa_fake_diff_conf1*all_diff_fake_aa_32 + (1-self.x_aa_fake_diff_conf1)*tf.image.resize_images(self.x_aa_fake, (size_32, size_32))
        
        self.loss_rec_warp1 = tf.reduce_mean(tf.square(tf.image.resize_images(self.Xb, (size_32, size_32)) - self.all_diffor_fake_b_32)) +\
                             tf.reduce_mean(tf.square(tf.image.resize_images(self.Xa, (size_32, size_32)) - self.all_diffor_fake_a_32)) +\
                             tf.reduce_mean(tf.square(tf.image.resize_images(self.Xa, (size_32, size_32)) - self.all_diffor_fake_aa_32)) 
 
        b_256_conf = self.x_b_fake_diff_conf3 
        a_256_conf = self.x_a_fake_diff_conf3 
        aa_256_conf = self.x_aa_fake_diff_conf3 
        
        self.diff_warp_fake_b_256 = all_diff_fake_b_128 
        self.all_diffor_fake_b_256 = b_256_conf* all_diff_fake_b_128 + (1-b_256_conf)*self.x_b_fake
        self.all_diffor_fake_a_256 = a_256_conf* all_diff_fake_a_128 + (1-a_256_conf)*self.x_a_fake
        self.all_diffor_fake_aa_256 = aa_256_conf* all_diff_fake_aa_128 + (1-aa_256_conf)*self.x_aa_fake
        
        
        self.loss_rec_warp = self.loss_rec_warp1 + self.loss_rec_warp2 + self.loss_rec_warp3
        self.log_vars.append(("loss_rec_pixel", self.loss_rec_warp)) 
        
        self.dis_real, self.dis_real_cls = self.discriminator(self.Xb)
        self.dis_f, self.dis_f_cls = self.discriminator(self.all_diffor_fake_b_128, reuse = True)
 
        
        def vgg16( inputs, reuse=False, scope='vgg_16'):
            with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                                    # outputs_collections=end_points_collection):
                    self.net1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    self.net1_p = slim.max_pool2d(self.net1, [2, 2], scope='pool1')
                    self.net2 = slim.repeat(self.net1_p, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    self.net2_p = slim.max_pool2d(self.net2, [2, 2], scope='pool2')
                    self.net3 = slim.repeat(self.net2_p, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    self.net3_p = slim.max_pool2d(self.net3, [2, 2], scope='pool3')
                    self.net4 = slim.repeat(self.net3_p, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    self.net4_p = slim.max_pool2d(self.net4, [2, 2], scope='pool4')
                    self.net5 = slim.repeat(self.net4_p, 3, slim.conv2d, 512, [3, 3], scope='conv5') 
                return self.net5, self.net2
        
        
        self.vgg_fb, self.vgg_fb2 =  vgg16(self.Xb)
        self.vgg_fb_f, self.vgg_fb_f2 =  vgg16(self.all_diffor_fake_b_128, reuse=True)
        self.vgg_fa, self.vgg_fa2 =  vgg16(self.Xa, reuse=True)
        self.vgg_fa_f, self.vgg_fa_f2 =  vgg16(self.x_a_fake, reuse=True)
#        self.loss_rec_f = tf.reduce_mean(tf.abs(self.vgg_fb -self.vgg_fb_f)) 
        self.loss_rec_f = 20*tf.reduce_mean(tf.abs(self.vgg_fb -self.vgg_fb_f))+tf.reduce_mean(tf.abs(self.vgg_fa -self.vgg_fa_f))
        self.log_vars.append(("loss_rec_f", self.loss_rec_f))
#        self.loss_rec_f2 = tf.reduce_mean(tf.abs(self.vgg_fb2 -self.vgg_fb_f2)) 
        self.loss_rec_f2 = 20*tf.reduce_mean(tf.abs(self.vgg_fb2 -self.vgg_fb_f2))+tf.reduce_mean(tf.abs(self.vgg_fa2 -self.vgg_fa_f2))
        self.log_vars.append(("loss_rec_f2", self.loss_rec_f2))
        t_vars = tf.global_variables() 
        self.vggnet_vars = [var for var in t_vars if 'vgg_16' in var.name]
        
        self.loss_VI = tf.reduce_mean(tf.abs(self.z_f -self.z_f_rec))
        self.log_vars.append(("loss_VI", self.loss_VI)) 
        self.loss_gen = loss_hinge_gen(self.dis_f)  
        self.log_vars.append(("loss_gen", self.loss_gen)) 
        self.loss_dis = loss_hinge_dis(self.dis_f, self.dis_real) 
#        self.loss_rec = tf.reduce_mean(tf.square(self.Xb - self.x_b_fake)) 

        self.loss_rec = 20*tf.reduce_mean(tf.square(self.Xb - self.x_b_fake)) + tf.reduce_mean(tf.square(self.Xa - self.x_a_fake)) + tf.reduce_mean(tf.square(self.Xa - self.x_aa_fake))
        self.log_vars.append(("loss_rec_pixel", self.loss_rec))    
 
        
#        self.loss_dis_ac = tf.reduce_mean(tf.square(self.dis_real_cls-self.Yb_matrix))+tf.reduce_mean(tf.square(self.dis_f_cls-self.Yb_matrix))
        self.loss_dis_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dis_real_cls, labels=self.Yb_matrix))
        self.log_vars.append(("loss_dis_ac", self.loss_dis_ac)) 
        self.loss_gen_ac = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dis_f_cls, labels=self.Yb_matrix))
        self.log_vars.append(("loss_gen_ac", self.loss_gen_ac))   
        
        self.loss_enc_ac0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.p_label_b, labels=self.Yb_matrix))
        self.log_vars.append(("loss_enc_ac0", self.loss_enc_ac0)) 
        self.loss_enc_ac1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.p_label_a, labels=self.Ya_matrix))
        self.log_vars.append(("loss_enc_ac1", self.loss_enc_ac1)) 
        self.loss_enc_ac = self.loss_enc_ac0 + self.loss_enc_ac1
        
        if self.is_wganGP == True: 
            epsilon_1 = tf.random_uniform([], 0.0, 1.0)
            interpolated = epsilon_1 * self.Xb + (1 - epsilon_1) * self.all_diffor_fake_b_128  
            self.D_logits = self.discriminator(interpolated,   reuse=True)[0]
            gradients = tf.gradients(self.D_logits, interpolated, name="D_logits_intp")[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0)) 
            interpolated = epsilon_1 * self.Xa + (1 - epsilon_1) * self.all_diffor_fake_a_128  
            self.D_logits = self.discriminator(interpolated,  reuse=True)[0]
            gradients = tf.gradients(self.D_logits, interpolated, name="D_logits_intp")[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            grad_penalty += tf.reduce_mean(tf.square(grad_l2 - 1.0)) 
            interpolated = epsilon_1 * self.Xa + (1 - epsilon_1) * self.all_diffor_fake_aa_128  
            self.D_logits = self.discriminator(interpolated,  reuse=True)[0]
            gradients = tf.gradients(self.D_logits, interpolated, name="D_logits_intp")[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            grad_penalty += tf.reduce_mean(tf.square(grad_l2 - 1.0)) 
            self.gp_loss_sum = self.log_vars.append(("grad_penalty", grad_penalty/3.0)) 
            self.loss_dis += self.gp_lambda * grad_penalty
        self.log_vars.append(("loss_dis", self.loss_dis))
        
        
        self.loss_kl1 = kl_loss(self.z_mu, self.z_log_var)
        self.loss_kl2 = kl_loss(self.z_mu_rec, self.z_log_var_rec)
        self.loss_kl = (self.loss_kl1 + self.loss_kl2)/2.0
        self.log_vars.append(("loss_kl", self.loss_kl))
        
         # C loss to classify c_enc_p 
        self.C_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Ya_matrix, logits = self.c_enc_p))
        self.C_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Yb_matrix, logits = self.c_enc_p_rec))
        self.C_loss = (self.C_loss1 + self.C_loss2)/2.0
        self.log_vars.append(("C_loss", self.C_loss))

        # adversarial loss
        self.adv_Y = tf.ones_like(self.c_enc_p, dtype=tf.float32) / tf.cast(self.y_dim, tf.float32)
        self.adv_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.adv_Y, logits=self.c_enc_p))
        self.adv_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.adv_Y, logits=self.c_enc_p_rec))
        self.adv_loss = (self.adv_loss1 + self.adv_loss2)/2.0
        self.log_vars.append(("adv_loss", self.adv_loss))

        # Optimizer
        self.enc_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.cls_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.dis_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)
        self.gen_opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_init, beta1=0.0, beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): 
            self.dis_solver = self.dis_opt.minimize(self.loss_dis + self.loss_dis_ac, var_list = self.discriminator.vars) 
            self.gen_solver = self.gen_opt.minimize(self.loss_gen + self.loss_rec*5 +\
                                                    self.loss_rec_warp*5 + \
                                                    5*self.loss_rec_f + \
                                                    self.loss_rec_f2  +self.loss_gen_ac, 
                                                    var_list = self.generator.vars )
            self.enc_solver = self.gen_opt.minimize(self.loss_gen + self.loss_VI + self.loss_rec*5 \
                                                    + self.loss_rec_warp*5 + \
                                                    5*self.loss_rec_f +\
                                                    self.loss_rec_f2  +self.loss_gen_ac + 0.1*self.loss_kl + self.adv_loss + self.loss_enc_ac, 
                                                    var_list = self.encoder.vars)
            self.cls_solver = self.cls_opt.minimize(self.C_loss, var_list = self.latent_discriminator.vars)

        # Summary
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)
        self.summary_op = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(max_to_keep=None)
        self.vggnet_saver = tf.train.Saver(self.vggnet_vars)

       
    def test(self, sample_folder, model_path, training_iters=50000, batch_size=64, n_dis = 1, n_samples = 16, restore=False):
        i = 0
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        if not os.path.exists(sample_folder + '/LPIPS'):
            os.makedirs(sample_folder + '/LPIPS')
        if not os.path.exists(sample_folder + '/FID'):
            os.makedirs(sample_folder + '/FID')
        self.sess.run(tf.global_variables_initializer())
        self.vggnet_saver.restore(self.sess, 'models/vgg_16.ckpt')
        for i in range(1):
            self.saver.restore(self.sess, model_path) 
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess)
            L1_error = []
            SSIM = []
            for iter in tqdm(range(training_iters)): 
                X_batch_a, Y_batch_a, X_batch_b, Y_batch_b = self.sess.run([self.img_batcha_test, self.label_batcha_test, self.img_batchb_test, self.label_batchb_test])
                X_batch_a = X_batch_a / 255.0 * 2 - 1
                X_batch_b = X_batch_b / 255.0 * 2 - 1
                samples, samples_rec = self.sess.run([self.all_diffor_fake_b_128, self.x_aa_fake], 
                                                     feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, \
                                                                self.Xb:X_batch_b, self.Yb:Y_batch_b, 
                                                                self.training:False})
                L1_v = np.mean(np.abs(((samples+1)/2)*255.0-((X_batch_b+1)/2)*255.0))
                L1_error.append(L1_v)
                ssim_v = ssim_score( (((samples+1)/2)*255.0).astype(np.uint8),  (((X_batch_b+1)/2)*255.0).astype(np.uint8))
                SSIM.append(ssim_v) 
                aa = np.concatenate([X_batch_a[0], X_batch_a[1], X_batch_a[2], X_batch_a[3]], 1)
                ss = np.concatenate([samples[0], samples[1], samples[2], samples[3]], 1)
                bb = np.concatenate([X_batch_b[0], X_batch_b[1], X_batch_b[2], X_batch_b[3]], 1)
                rr = np.concatenate([samples_rec[0], samples_rec[1], samples_rec[2], samples_rec[3]], 1)
                allall = np.concatenate([aa, ss, bb, rr], 0)
                plt.imsave('{}/LPIPS/{}_paired_test.png'.format(sample_folder, str(iter).zfill(3)), ((allall+1)/2*255.0).astype('uint8'))
                for bb in range(batch_size):
                    plt.imsave('{}/FID/{}_{}_paired_test.png'.format(sample_folder, str(iter).zfill(3), str(bb).zfill(3)), ((samples[bb]+1)/2*255.0).astype('uint8'))
            print 'models_{} L1 error: {}'.format(i, np.mean(np.array(L1_error)))
            print 'models_{} SSIM error {}:'.format(i, np.mean(np.array(SSIM)))
            
    
    def train(self, sample_folder, training_iters=50000, batch_size=64, n_dis = 1, n_samples = 16, restore=False):
        i = 0
        self.sess.run(tf.global_variables_initializer())
        self.vggnet_saver.restore(self.sess, 'models/vgg_16.ckpt')
        restore_iter = 0
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                print 'Restoring from {}...'.format(ckpt.model_checkpoint_path),
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[1]
                restore_iter = int(stem.split('-')[-1])
                i = restore_iter / 500
                #self.sess.run(self.global_step.assign(restore_iter))
                print 'done'
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess)
        for iter in range(restore_iter, training_iters):
            X_batch_a, Y_batch_a, X_batch_b, Y_batch_b = self.sess.run([self.img_batcha, self.label_batcha, self.img_batchb, self.label_batchb])
            X_batch_a = X_batch_a / 255.0 * 2 - 1
            X_batch_b = X_batch_b / 255.0 * 2 - 1  
            for _ in range(n_dis):
                self.sess.run( self.dis_solver , feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b,
                                                            self.z_in: sample_z(batch_size, self.in_dim),
                                                            self.training:True})
                self.sess.run(self.cls_solver, feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b,
                                                            self.z_in: sample_z(batch_size, self.in_dim),
                                                            self.training:True})
            for _ in range(1):
                self.sess.run([self.gen_solver, self.enc_solver],feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b,  
                                                         self.z_in: sample_z(batch_size, self.in_dim),
                                                         self.training:True})
            # summary
            summary_str = self.sess.run(self.summary_op,feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b, 
                                                                   self.z_in: sample_z(batch_size, self.in_dim),
                                                                   self.training:False})
            self.summary_writer.add_summary(summary_str, iter)

             #print loss
            if iter % 100 == 0 or iter < 100:
                fetch_list = [self.loss_VI, self.loss_rec, self.loss_rec_f, self.loss_rec_f2, self.loss_gen, self.loss_dis, self.loss_dis_ac, self.loss_rec_warp]
                loss_VI_curr, loss_rec_curr, loss_rec_f_curr, loss_rec_f2_curr,\
                loss_gen_curr, loss_dis_curr, loss_dis_ac_curr, loss_rec_warp_curr = \
                self.sess.run(fetch_list, feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b, 
                                               self.z_in: sample_z(batch_size, self.in_dim),
                                               self.training:False})

                print('TRAIN Iter: {};  loss_vi: {:.4}; loss_rec: {:.4}; loss_rec_f: {:.4}; loss_rec_f2: {:.4}; loss_gen: {:.4}; \
                      loss_dis: {:.4};  loss_dis_ac: {:.4}; loss_rec_warp: {:.4}'.
                      format(iter, loss_VI_curr,\
                             loss_rec_curr, \
                             loss_rec_f_curr, \
                             loss_rec_f2_curr, \
                             loss_gen_curr, \
                             loss_dis_curr, \
                             loss_dis_ac_curr, \
                             loss_rec_warp_curr))


                if iter % 500 == 0:
                    X_batch_a, Y_batch_a, X_batch_b, Y_batch_b = self.sess.run([self.img_batcha_test, self.label_batcha_test, self.img_batchb_test, self.label_batchb_test])
                    
                    X_batch_a = X_batch_a / 255.0 * 2 - 1
                    X_batch_b = X_batch_b / 255.0 * 2 - 1  
                    fetch_list = [self.loss_VI, self.loss_rec, self.loss_rec_f, self.loss_rec_f2, self.loss_gen, self.loss_dis, self.loss_dis_ac, self.loss_rec_warp ]
                    loss_VI_curr, loss_rec_curr, loss_rec_f_curr, loss_rec_f2_curr, loss_gen_curr, loss_dis_curr, loss_dis_ac_curr, loss_rec_warp_curr  = self.sess.run(fetch_list,
                                        feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b, 
                                                   self.z_in: sample_z(batch_size, self.in_dim),
                                                   self.training:False})
    
                    print('TESt Iter: {}; loss_vi: {:.4}; loss_rec: {:.4}; loss_rec_f: {:.4}; loss_rec_f2: {:.4}; loss_gen: {:.4};\
                          loss_dis: {:.4}; loss_dis_ac: {:.4}; loss_rec_warp: {:.4}; '.
                      format(iter, loss_VI_curr, loss_rec_curr, loss_rec_f_curr, loss_rec_f2_curr, loss_gen_curr, loss_dis_curr, \
                             loss_dis_ac_curr, loss_rec_warp_curr))
                    samples = self.sess.run(self.all_diffor_fake_b_128, feed_dict={self.Xa:X_batch_a, self.Ya:Y_batch_a, self.Xb:X_batch_b, self.Yb:Y_batch_b, 
                                                                      self.z_in: sample_z(batch_size, self.in_dim),
                                                                      self.training:False})
                    print 'TEST L1 error:{};'.format( np.mean(np.abs(((samples+1)/2)*255.0-((X_batch_b+1)/2)*255.0)))
                    aa = np.concatenate([X_batch_a[0], X_batch_a[1], X_batch_a[2], X_batch_a[3]], 1)
                    ss = np.concatenate([samples[0], samples[1], samples[2], samples[3]], 1)
                    bb = np.concatenate([X_batch_b[0], X_batch_b[1], X_batch_b[2], X_batch_b[3]], 1)
                    allall = np.concatenate([aa, ss, bb], 0)
                    
                    plt.imsave('{}/{}_paired_test.png'.format(sample_folder, str(i).zfill(3)), ((allall+1)/2*255.0).astype('uint8'))
          
                    i += 1

                if (iter % 1000 == 0) or iter == training_iters - 1:
                    save_path = self.model_dir + "model.ckpt"
                    self.saver.save(self.sess, save_path, global_step=iter)
        self.sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_dim', type=int, default=62, help='label dimension')
    parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--sample_folder', type=str, default='Samples/IDUnet', help='save images')
    parser.add_argument('--log_dir', type=str, default='logs/IDUnet', help='save logs')
    parser.add_argument('--model_dir', type=str, default='models/IDUnet/', help='save models')
    parser.add_argument('--mode', default='training', choices=['training', 'generation'])
    parser.add_argument('--training_iters', type=int, default=100000, help='MAX training iters')
    parser.add_argument('--model_path', type=str, default=None, help='reload model path')
    parser.add_argument('--image_path', type=str, default='datasets/generation', help='generation image path')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sample_folder = args.sample_folder
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    y_dim = args.y_dim
    y_diff_dim = y_dim 
    encoder = encoder_style_SPADE_VI(y_num = 1, y_dim=y_dim, channel=32) 
    discriminator = Discriminator128_AC(y_num = 1, y_dim=y_dim) 
    generator = generate_style_res_VI(y_num = 1, y_dim=y_dim, y_diff_dim = y_diff_dim, channel=32)
    latent_discriminator = LatentDiscriminator(y_dim = y_dim)  
    batch_size = args.batch_size
    train_batch_size = args.train_batch_size

    gan = SNGAN(encoder, generator, discriminator, latent_discriminator, batch_size, train_batch_size, log_dir=args.log_dir, model_dir=args.model_dir)
    if args.mode=='training':
        gan.train(sample_folder, batch_size=batch_size, training_iters=400000, restore = False)
    elif args.mode=='test': 
        gan.test(sample_folder+'/test', model_path=args.model_path, batch_size=batch_size, training_iters=5560/batch_size, restore = False)
 