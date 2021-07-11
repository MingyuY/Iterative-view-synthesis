import tensorflow.contrib.layers as tcl
from layers.ops import *
import tensorflow as tf
        
def VGG16(inputs, out_dim, embs, is_training = True, isvae = False):
    
    with tf.name_scope('conv1_1') as scope:
        out = tcl.conv2d(inputs, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training':is_training})
    with tf.name_scope('conv1_2') as scope:
        out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    out = tcl.max_pool2d(out, kernel_size=2, stride=2)
    
    with tf.name_scope('conv2_1') as scope:
        out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv2_2') as scope:
        out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    out = tcl.max_pool2d(out, kernel_size=2, stride=2)

    with tf.name_scope('conv3_1') as scope:
        out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv3_2') as scope:
        out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv3_3') as scope:
        out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    out = tcl.max_pool2d(out, kernel_size=2, stride=2)

    with tf.name_scope('conv4_1') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv4_2') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv4_3') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    out = tcl.max_pool2d(out, kernel_size=2, stride=2)

    with tf.name_scope('conv5_1') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv5_2') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    with tf.name_scope('conv5_3') as scope:
        out = tcl.conv2d(out, num_outputs=512, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                         normalizer_fn=tcl.batch_norm,
                         normalizer_params={'scale': True, 'is_training': is_training})
    # out = tcl.max_pool2d(out, kernel_size=2, stride=2)

    out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)

    out = tcl.flatten(out)

    with tf.name_scope('fc6') as scope:
        out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                  normalizer_params={'scale': True, 'is_training': is_training})
        # out = tcl.dropout(out, keep_prob=keep_prob, is_training=is_training)

    # with tf.name_scope('fc7') as scope:
    #     out = tcl.fully_connected(out, 4096, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
    #                               normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.dropout(out, keep_prob=keep_prob, is_training=is_training)
    if isvae:
        with tf.name_scope('fc8') as scope:
            out1 = tcl.fully_connected(out, out_dim, activation_fn=None)
            out2 = tcl.fully_connected(out, out_dim, activation_fn=None)
        return out1, out2

    else:
        with tf.name_scope('fc8') as scope:
            out = tcl.fully_connected(out, out_dim, activation_fn=None)

        return out
    
def vgg_layer():
    import os
    import numpy as np
    import tensorflow as tf
    slim = tf.contrib.slim
    PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd())) 
    
    tf.app.flags.DEFINE_string('pretrained_model_path',  os.path.join(PROJECT_PATH, 'vgg_16.ckpt'), '')
    FLAGS = tf.app.flags.FLAGS
    
    def vgg_arg_scope(weight_decay=0.1):
      """  VGG arg scope.
      Args:
        weight_decay: The l2 regularization coefficient.
      Returns:
        An arg_scope.
      """
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
          return arg_sc
    
    def vgg16(inputs,scope='vgg_16'):
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                                # outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                # net = slim.max_pool2d(net, [2, 2], scope='pool5')
                # net = slim.fully_connected(net, 4096, scope='fc6')
                # net = slim.dropout(net, 0.5, scope='dropout6')
                # net = slim.fully_connected(net, 4096, scope='fc7')
                # net = slim.dropout(net, 0.5, scope='dropout7')
                # net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
            return net
    
    def net(shape):
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        with tf.variable_scope('vgg_16', 'vgg_16', [input_image]) as sc:
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],):
                                # outputs_collections=end_points_collection):
                net1 = slim.repeat(input_image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net1_p = slim.max_pool2d(net1, [2, 2], scope='pool1')
                net2 = slim.repeat(net1_p, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net2_p = slim.max_pool2d(net2, [2, 2], scope='pool2')
                net3 = slim.repeat(net2_p, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net3_p = slim.max_pool2d(net3, [2, 2], scope='pool3')
                net4 = slim.repeat(net3_p, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net4_p = slim.max_pool2d(net4, [2, 2], scope='pool4')
                net5 = slim.repeat(net4_p, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#        with slim.arg_scope(vgg_arg_scope()):
#            conv5_3 = vgg16(input_image)  
    
        init = tf.global_variables_initializer() 
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
        with tf.Session() as sess:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None: 
                variable_restore_op(sess)
            a = sess.run([net5],feed_dict={input_image:np.arange(360000).reshape(1,300,400,3)})
#    
#    if __name__ == '__main__':
#        net()
#        print(tf.trainable_variables())




class VggBlock(object):
    def __init__(self,
                 z_dim = 256,
                 y_num = 40,
                 y_dim = 2):
        self.name = 'VggBlock'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
                                 normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)

        w_outs = []
        for i in range(40):
            emb = embs[:, i, :]  # (N, c)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
                         normalizer_fn=None)
        out = out + inputs
        return out

    def __call__(self, inputs, labels, embs, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att1") as scope:
                out = self.do_att(out, embs1)

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att2") as scope:
                out = self.do_att(out, embs2)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att3") as scope:
                out = self.do_att(out, embs3)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att4") as scope:
                out = self.do_att(out, embs4)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Encoder128_block(object):
    def __init__(self,
                 z_dim = 256,
                 y_num = 40,
                 y_dim = 2):
        self.name = 'Encoder128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def __call__(self, inputs, labels, out_sh, emb_matrix, is_training = True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            out = tcl.flatten(out_sh)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            hidden = out
            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]

            # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            weights = tf.matmul(out, emb_y)  # [N,4,40]
            weight = tf.reduce_mean(weights, axis=2, keep_dims=False)  # [N,4]
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            hout = out * weight  # [N,4,z_dim]
            out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Discriminator128_block(object):
    def __init__(self):
        self.name = 'Discriminator128'
        self.channel = 32

    def __call__(self, inputs, labels, out_sh, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)

            out = tf.reduce_sum(out_sh, axis=(1, 2))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')

            # emb_y = fully_connected(labels, self.channel * 8, sn=True, scope='y_proj')
            # proj = tf.reduce_sum(emb_y * out, axis=1, keep_dims=True)
            # y_onehot = tf.one_hot(y, num_classes)
            # y_proj = fully_connected(y_onehot, 1024, sn = true, scope='y_proj')
            # y_proj = tf.reduce_sum(y_proj * out, axis=1, keep_dims=True)


            pros = []
            for i in range(40):
                w_y = labels[:, i, :]
                emb_y = fully_connected(w_y, self.channel * 8, use_bias=False, sn=True, scope='y_proj' + str(i))
                pro = tf.reduce_sum(emb_y * out, axis=1, keepdims=True)
                pros.append(pro)
            pros = tf.concat(pros, axis=1)
            y_proj = tf.reduce_sum(pros, axis=1, keepdims=True)

            #            q = tcl.fully_connected(out, 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm,
            #                                      normalizer_params={'scale': True, 'is_training': is_training})
            #
            #            q = tcl.fully_connected(q, self.z_dim, activation_fn=None)

            return d + y_proj * 0.025

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Encoder128(object):
    def  __init__(self,
                 z_dim = 256,
                 y_num = 40,
                 y_dim = 2,
                 f_num = 2):
        self.name = 'Encoder128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        
    # def do_att(self, inputs, embs, is_training = True):
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1]*input_shape[2], 32))  #(N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(self.y_num):
    #         emb = embs[:,i,:]   #(N, c)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  #(N, c, 1)
    #         # emb = tf.expand_dims(emb, 2)  #(N, c, 1)
    #         att = tf.matmul(input_r, emb)  #(N, size**2, 1)
    #         #att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))   #(N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  #(N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)   #(N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out

    def do_att(self, inputs, embs, is_training=True):#multi
        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            att_weights = tf.nn.softmax(tf.squeeze(att))  # (40, N, size**2)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(att_weights, (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def do_cbn(self, inputs, atts, is_training = True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)
        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # adain_x = tf.reduce_sum(adains, axis=0)
        return out

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_params=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1,0,2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  #[40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     for i in range(self.y_num):
    #         att_weight = att_weights[i,:,:,:]  # (1, N, size**2, 1)
    #         att_weight = tf.squeeze(att_weight)  #(N, size**2)
    #         # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #                      normalizer_params=None)
    #     out = out + inputs
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training = True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            #[embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs 
            # [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training':is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att1") as scope:
                # out = self.do_att(out, embs1)
                att1 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att1, is_training=is_training)
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            
                
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att2") as scope:
                # out = self.do_att(out, embs2)
                att2 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att2, is_training=is_training)
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)
                
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att3") as scope:
                # out = self.do_att(out, embs3)
                att3 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att3, is_training=is_training)
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)
                
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att4") as scope:
                # out = self.do_att(out, embs4)
                att4 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att4, is_training=is_training)
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)
               
#            with tf.name_scope('conv5_1') as scope:
#                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})
#            with tf.name_scope('conv5_2') as scope:
#                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})
#            with tf.name_scope('conv5_3') as scope:
#                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})
#            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
        
#            out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)
#            with tf.name_scope("att5") as scope:
#                out = self.do_att(out, embs5)
        
            out = tcl.flatten(out)
        
            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
                                          
#            hidden = out
#            embs_matrix = tf.nn.embedding_lookup(embs_matrix, labels)
#            embs_matrix = tf.reshape(embs_matrix, (-1, self.z_dim, 1))
#            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1]/self.z_dim, self.z_dim))
#            weight = tf.matmul(out, embs_matrix)
#            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1]/self.z_dim))
#            weight = tf.nn.softmax(weight)
#            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1]/self.z_dim, 1))
#            out = out * weight
#            out = tf.reduce_sum(out, axis = 1)
#             hidden = out
#             out = tf.reshape(out, (-1, hidden.shape.as_list()[-1]/self.z_dim, self.z_dim))  #[N,4,z_dim]
#
#             # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
#             #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
#             # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
#             y_reshape = tf.transpose(labels, (1,0,2))  # [40,N,2]
#             emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
#             emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale = True)
#             weights = tf.matmul(out, emb_y)   #[N,4,40]
#             weight = tf.reduce_mean(weights, axis=2, keep_dims=False)   #[N,4]
#             weight = tf.nn.softmax(weight)
#             weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))   #[N,4,1]
#             hout = out * weight   #[N,4,z_dim]
#             out = tf.reduce_sum(hout, axis=1)

            
            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128(object):
    def __init__(self,
                 z_dim = 256,
                 y_num = 40,
                 y_dim = 2,
                 f_num = 2):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    # def do_att(self, inputs, embs, is_training = True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1]*input_shape[2], 32))  #(N, size**2, c)
    #
    #     w_outs = []
    #     att_w = []
    #     for i in range(self.y_num):
    #         emb = embs[:,i,:]   #(N, c)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  #(N, c, 1)
    #         # emb = tf.expand_dims(emb, 2)  #(N, c, 1)
    #         att = tf.matmul(input_r, emb)  #(N, size**2, 1)
    #         #att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))   #(N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  #(N, size, size, c)
    #         w_outs.append(w_out)
    #         att_w.append(att_weight)
    #     w_outs = tf.concat(w_outs, axis=3)   #(N, size, size, 40*c)
    #     att_w = tf.concat(att_w, axis=3)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out, att_w

    def do_att(self, inputs, embs, is_training=True):#multi
        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            att_weights = tf.nn.softmax(tf.squeeze(att))  # (40, N, size**2)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(att_weights, (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def do_cbn(self, inputs, atts, is_training = True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)
        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # adain_x = tf.reduce_sum(adains, axis=0)
        return out

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #                              normalizer_params=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num, 1, 1, 1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1, 0, 2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  # [40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     for i in range(self.y_num):
    #         att_weight = att_weights[i, :, :, :]  # (1, N, size**2, 1)
    #         att_weight = tf.squeeze(att_weight)  # (N, size**2)
    #         # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         emb = tf.squeeze(embs[i, :, :, :])  # (N,c)
    #         emb = tf.reshape(emb, (-1, 1, 1, 32))  # (N,1,1,c)
    #         emb = tf.tile(emb, (1, input_shape[1], input_shape[2], 1))  # (N, size, size, c)
    #         w_out = tf.multiply(emb, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_out = tf.reduce_mean(w_outs, axis=0)  # (N, size, size, 32)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_params=None)
    #     out = out + down_inputs
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training = True, reuse = False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            #z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            # out = tf.concat([inputs, labels], 1)
            w = 4
#            [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
#             [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope('fc7') as scope:
            #     out = tcl.fully_connected(out, 4096, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            
#            embs_matrix = tf.nn.embedding_lookup(embs_matrix, labels)
#            embs_matrix = tf.reshape(embs_matrix, (-1, self.z_dim, 1))
#            out = tf.reshape(out, (-1, 1024/self.z_dim, self.z_dim))
#            weight = tf.matmul(out, embs_matrix)
#            weight = tf.reshape(weight, (-1, 1024/self.z_dim))
#            weight = tf.nn.softmax(weight)
#            weight = tf.reshape(weight, (-1, 1024/self.z_dim, 1))
#            out = out * weight
#            out = tf.reduce_sum(out, axis = 1)
#             hidden = out
#             out = tf.reshape(out, (-1, hidden.shape.as_list()[-1]/self.z_dim, self.z_dim))  #[N,4,z_dim]
#
#             # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
#             #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
#             # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
#             y_reshape = tf.transpose(labels, (1,0,2))  # [40,N,2]
#             emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
#             emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale = True)
#             weights = tf.matmul(out, emb_y)   #[N,4,40]
#             weight = tf.reduce_mean(weights, axis=2)   #[N,4]
#             weight = tf.nn.softmax(weight)
#             weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))   #[N,4,1]
#             hout = out * weight   #[N,4,z_dim]
#             out = tf.reduce_sum(hout, axis=1)
            
            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w*w*256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256)) #4*4*512
#            with tf.name_scope("att5") as scope:
#                out = self.do_att(out, embs5)
            
#            with tf.name_scope('conv5_3') as scope:
#                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})
#            with tf.name_scope('conv5_2') as scope:
#                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})
#            with tf.name_scope('conv5_1') as scope:
#                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
#                                 normalizer_fn=tcl.batch_norm,
#                                 normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            with tf.name_scope("att4") as scope:
                # out = self.do_att(out, embs4)
                att4 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att4, is_training=is_training)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope("att4") as scope:
            #     out, att4 = self.do_att(out, embs)
            # out = self.do_adain(out, att4)
            with tf.name_scope("att3") as scope:
                # out = self.do_att(out, embs3)
                att3 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att3, is_training=is_training)
                
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope("att3") as scope:
            #     out, att3 = self.do_att(out, embs)
            # out = self.do_adain(out, att3)
            with tf.name_scope("att2") as scope:
                # out = self.do_att(out, embs2)
                att2 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att2, is_training=is_training)
                
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope('conv2_0') as scope:
            #     out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                      normalizer_fn=tcl.batch_norm,
            #                      normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope("att2") as scope:
            #     out, att2 = self.do_att(out, embs)
            # out = self.do_adain(out, att2)
            with tf.name_scope("att1") as scope:
                # out = self.do_att(out, embs1)
                att1 = self.do_att(out, embs, is_training=is_training)
            out = self.do_cbn(out, att1, is_training=is_training)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})

            # with tf.name_scope("att1") as scope:
            #     out, att1 = self.do_att(out, embs)
            # out = self.do_adain(out, att1)
            # with tf.name_scope('conv0_1') as scope:
            #     out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128_new(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
        #                          normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)

        w_outs = []
        for i in range(self.y_num):
            emb = embs[:, i, :]  # (N, c)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
            # emb = tf.expand_dims(emb, 2)  #(N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
        #                  normalizer_params=None)
        out = out + inputs
        return out

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #                              normalizer_params=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num, 1, 1, 1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1, 0, 2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  # [40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     for i in range(self.y_num):
    #         att_weight = att_weights[i, :, :, :]  # (1, N, size**2, 1)
    #         att_weight = tf.squeeze(att_weight)  # (N, size**2)
    #         # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         emb = tf.squeeze(embs[i, :, :, :])  # (N,c)
    #         emb = tf.reshape(emb, (-1, 1, 1, 32))  # (N,1,1,c)
    #         emb = tf.tile(emb, (1, input_shape[1], input_shape[2], 1))  # (N, size, size, c)
    #         w_out = tf.multiply(emb, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_out = tf.reduce_mean(w_outs, axis=0)  # (N, size, size, 32)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_params=None)
    #     out = out + down_inputs
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            # out = tf.concat([inputs, labels], 1)
            w = 4
            #            [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
                # with tf.name_scope('fc7') as scope:
                #     out = tcl.fully_connected(out, 4096, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                #                               normalizer_params={'scale': True, 'is_training': is_training})

            #            embs_matrix = tf.nn.embedding_lookup(embs_matrix, labels)
            #            embs_matrix = tf.reshape(embs_matrix, (-1, self.z_dim, 1))
            #            out = tf.reshape(out, (-1, 1024/self.z_dim, self.z_dim))
            #            weight = tf.matmul(out, embs_matrix)
            #            weight = tf.reshape(weight, (-1, 1024/self.z_dim))
            #            weight = tf.nn.softmax(weight)
            #            weight = tf.reshape(weight, (-1, 1024/self.z_dim, 1))
            #            out = out * weight
            #            out = tf.reduce_sum(out, axis = 1)

            # hidden = out
            # out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]
            #
            # # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            # #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            # y_reshape = tf.transpose(labels, (1, 0, 2))  # [40,N,2]
            # emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            # emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            # weights = tf.matmul(out, emb_y)  # [N,4,40]
            # weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            # weight = tf.nn.softmax(weight)
            # weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            # hout = out * weight  # [N,4,z_dim]
            # out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            # with tf.name_scope("att4") as scope:
            #     out = self.do_att(out, embs4)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (8,8))  #8*8*256
            with tf.name_scope("att4") as scope:
                out = self.do_att(out, embs4)
            # with tf.name_scope("att3") as scope:
            #     out = self.do_att(out, embs3)

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (16, 16))  #16*16*128
            with tf.name_scope("att3") as scope:
                out = self.do_att(out, embs3)
            # with tf.name_scope("att2") as scope:
            #     out = self.do_att(out, embs2)

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (32, 32))  # 32*32*64
            with tf.name_scope("att2") as scope:
                out = self.do_att(out, embs2)
            # with tf.name_scope("att1") as scope:
            #     out = self.do_att(out, embs1)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (64, 64))
            with tf.name_scope("att1") as scope:
                out = self.do_att(out, embs1)
            with tf.name_scope('conv1') as scope:
                out = tcl.conv2d(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Encoder128_concat(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2):
        self.name = 'Encoder128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def __call__(self, inputs, labels, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            out = tf.concat([inputs, y], 3)

            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128_concat(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2,
                 f_num=2):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def __call__(self, inputs, labels, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            out = tf.concat([inputs, labels], 1)
            w = 4
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            # with tf.name_scope('conv2_0') as scope:
            #     out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                      normalizer_fn=tcl.batch_norm,
            #                      normalizer_params={'scale': True, 'is_training': is_training})


            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Encoder128_att4(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2,
                 f_num=4):
        self.name = 'Encoder128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
    #     #                          normalizer_fn=None)
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #                              normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1,0,2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  #[40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     for i in range(self.y_num):
    #         att_weight = att_weights[i,:,:,:]  # (1, N, size**2, 1)
    #         att_weight = tf.squeeze(att_weight)  #(N, size**2)
    #         # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         emb = tf.squeeze(embs[i,:,:,:])  # (N,c)
    #         emb = tf.reshape(emb, (-1,1,1,32))  # (N,1,1,c)
    #         emb = tf.tile(emb, (1, input_shape[1], input_shape[2], 1))  # (N, size, size, c)
    #         w_out = tf.multiply(emb, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     out = tf.reduce_sum(w_outs, axis=0)  # (N, size, size, c)
    #     # w_outs = tf.concat(w_outs, axis=3)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_params=None)
    #     out = out + down_inputs
    #     return out

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1,0,2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  #[40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att_weights), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))
    #     return att_w_soft

    def do_adain(self, inputs, atts, is_training = True):#conditional bn
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)
        adains = []

        for i in range(self.y_num):
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            # att = tf.reshape(atts[0,:,:,:,i], (-1,input_shape[1]**2))  # (N, size,size)
            # mu_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            # sigma_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tf.reduce_mean(adains, axis=0)
        return out

    # def do_adain(self, inputs, atts, is_training = True):#conditional bn
    #     down_inputs = tcl.conv2d(inputs, num_outputs=self.f_num, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                              normalizer_params={'scale': True, 'is_training': is_training})
    #     input_shape = inputs.shape.as_list()
    #     mu_x, sigma_x = tf.nn.moments(down_inputs, axes=[0,1,2], keep_dims=True)
    #     x_ = (down_inputs-mu_x)/(sigma_x+1e-6)
    #     adains = []
    #     atts = tf.transpose(atts, (4,1,2,3,0))  # (40, N, size,size, f_num)
    #     for i in range(self.y_num):
    #         # att = tf.expand_dims(atts[:,:,:,i],3)
    #         # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
    #         att = atts[0,:,:,:,:]  # (N, size,size, f_num)
    #         # sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
    #         mu_y, sigma_y = tf.nn.moments(att, axes=[1, 2], keep_dims=True)  # (1, 1, 1, c)
    #         adain = sigma_y*x_+mu_y
    #         adains.append(adain)
    #     adain_x = tf.concat(adains, axis=3)
    #     out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tf.reduce_mean(adains, axis=0)
    #     return out

    def do_att(self, inputs, embs, is_training=True):#multi attngan attention

        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def use_noise(self, inputs, noise_input, layer):
        input_shape = inputs.shape.as_list()
        weight = tf.get_variable('weight%d'%layer, shape = [input_shape[-1]],  dtype=tf.float32, initializer=tf.initializers.zeros())
        noise = noise_input*tf.reshape(weight, [1,1,1,-1])
        x = inputs + noise
        return x

    # def do_att(self, inputs, embs, is_training=True):#multi  my attention
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #     att_w = []
    #     for i in range(self.f_num):
    #         input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #         emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
    #         emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
    #         emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
    #         att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
    #         att_weights = tf.nn.softmax(tf.reshape(att, (self.y_num, -1, input_shape[1]**2)))  # (40, N, size**2)
    #         # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
    #         att_w_soft = tf.reshape(tf.transpose(att_weights, (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
    #         att_w.append(tf.expand_dims(att_w_soft,0))
    #     att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
    #     # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
    #     return att_w

    # def do_adain(self, inputs, atts, is_training=True):
    #     input_shape = inputs.shape.as_list()
    #     mu_x, sigma_x = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    #     x_ = (inputs - mu_x) / (sigma_x + 1e-6)  #[N, 1, 1, c]
    #     adains = []
    #
    #     for i in range(self.y_num):
    #         att = atts[:, :, :, i, :]  #[N, H, W, 4]
    #         mu_y, sigma_y = tf.nn.moments(att, axes=[1, 2], keep_dims=True) #[N, 1, 1, 4]
    #         group_adains = []
    #         g_len = input_shape[-1]/self.f_num
    #         for k in range(self.f_num):
    #             group_adain = tf.expand_dims(sigma_y[:,:,:,k],3) * x_[:,:,:,k*g_len:(k+1)*g_len] + tf.expand_dims(mu_y[:,:,:,k],3)
    #             group_adains.append(group_adain)
    #         adain = tf.concat(group_adains, axis=3)
    #         adains.append(adain)
    #     adain_x = tf.concat(adains, axis=3)
    #     out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tf.reduce_sum(adains, axis=0)
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, bs, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            # [embs1, embs2, embs3, embs4] = embs
            noise_inputs = []
            for i in range(4):
                size = 2**(3-i)*4
                shape = [bs, size, size, 1]
                noise_input = tf.get_variable('noise%d'%i, shape=shape, initializer=tf.initializers.random_normal(),
                                              trainable=False)
                noise_inputs.append(noise_input)

            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.use_noise(out, noise_inputs[0], 0)
            with tf.name_scope("att1") as scope:
                att1 = self.do_att(out, embs, is_training=is_training)
                # out = self.do_att(out, embs1)
            out = self.do_adain(out, att1, is_training=is_training)

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.use_noise(out, noise_inputs[1], 1)
            with tf.name_scope("att2") as scope:
                # out = self.do_att(out, embs2)
                att2 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att2, is_training=is_training)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.use_noise(out, noise_inputs[2], 2)
            with tf.name_scope("att3") as scope:
                # out = self.do_att(out, embs3)
                att3 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att3, is_training=is_training)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.use_noise(out, noise_inputs[3], 3)
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att4") as scope:
                # out = self.do_att(out, embs4)
                att4 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att4, is_training=is_training)

            # with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            #            out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            # hidden = out
            # out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]
            #
            # # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            # #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            # emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            # emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            # weights = tf.matmul(out, emb_y)  # [N,4,40]
            # weight = tf.reduce_mean(weights, axis=2, keep_dims=False)  # [N,4]
            # weight = tf.nn.softmax(weight)
            # weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            # hout = out * weight  # [N,4,z_dim]
            # out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)

            # with tf.name_scope('fc8') as scope:
            #     out = tcl.fully_connected(out, 160, activation_fn=None)
            # z_mu = []
            # z_logvar = []
            # for i in range(self.y_num):
            #     vec = out[:,i*4:(i+1)*4]
            #     mu = vec[:,:2]
            #     sigma = vec[:,2:]
            #     z_mu.append(mu)
            #     z_logvar.append(sigma)

            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128_att4(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2,
                 f_num=4):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
    #     #                          normalizer_fn=None)
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #                              normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1,0,2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  #[40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att_weights), (1, 2, 0)),
    #                             (-1, input_shape[1], input_shape[2], self.y_num))
    #     for i in range(self.y_num):
    #         att_weight = att_weights[i,:,:,:]  # (1, N, size**2, 1)
    #         att_weight = tf.squeeze(att_weight)  #(N, size**2)
    #         # att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1]**2)))   #(N, size**2)
    #         # att_weight = tf.nn.sigmoid(att)   #(N, size**2, 1)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         emb = tf.squeeze(embs[i,:,:,:])  # (N,c)
    #         emb = tf.reshape(emb, (-1,1,1,32))  # (N,1,1,c)
    #         emb = tf.tile(emb, (1, input_shape[1], input_shape[2], 1))  # (N, size, size, c)
    #         w_out = tf.multiply(emb, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     out = tf.reduce_sum(w_outs, axis=0)  # (N, size, size, c)
    #     # w_outs = tf.concat(w_outs, axis=3)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_params=None)
    #     out = out + down_inputs
    #     return out

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
    #     #                          normalizer_fn=None)
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #     embs = tf.transpose(embs, (1,0,2))  # [N,40,c]->[40,N,c]
    #     embs = tf.expand_dims(embs, 3)  #[40,N,c,1]
    #     att = tf.matmul(input_r_, embs)  # (40, N, size**2, 1)
    #     att_weights = tf.nn.softmax(att, axis=0)
    #     att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att_weights), (1, 2, 0)),
    #                             (-1, input_shape[1], input_shape[2], self.y_num))
    #
    #     return att_w_soft

    def do_adain(self, inputs, atts, is_training = True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)
        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            # att = tf.reshape(atts[0, :, :, :, i], (-1, input_shape[1] ** 2))  # (N, size,size)
            # mu_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                            normalizer_params={'scale': True, 'is_training': is_training})
            # sigma_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tf.reduce_mean(adains, axis=0)
        return out

    # def do_adain(self, inputs, atts, is_training = True):#conditional bn
    #     down_inputs = tcl.conv2d(inputs, num_outputs=self.f_num, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                              normalizer_params={'scale': True, 'is_training': is_training})
    #     input_shape = inputs.shape.as_list()
    #     mu_x, sigma_x = tf.nn.moments(down_inputs, axes=[1,2], keep_dims=True)
    #     x_ = (down_inputs-mu_x)/(sigma_x+1e-6)
    #     adains = []
    #     atts = tf.transpose(atts, (4,1,2,3,0))  # (40, N, size,size, f_num)
    #     for i in range(self.y_num):
    #         # att = tf.expand_dims(atts[:,:,:,i],3)
    #         # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
    #         att = atts[0,:,:,:,:]  # (N, size,size, f_num)
    #         # sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
    #         mu_y, sigma_y = tf.nn.moments(att, axes=[1, 2], keep_dims=True)  # (1, 1, 1, c)
    #         adain = sigma_y*x_+mu_y
    #         adains.append(adain)
    #     adain_x = tf.concat(adains, axis=3)
    #     out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1,  normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tf.reduce_mean(adains, axis=0)
    #     return out

    def do_att(self, inputs, embs, is_training=True):#multi

        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def use_noise(self, inputs, noise_input, layer):
        input_shape = inputs.shape.as_list()
        weight = tf.get_variable('weight%d'%layer, shape = [input_shape[-1]],  dtype=tf.float32, initializer=tf.initializers.zeros())
        noise = noise_input*tf.reshape(weight, [1,1,1,-1])
        x = inputs + noise
        return x

    # def do_att(self, inputs, embs, is_training=True):#multi  my attention
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #     att_w = []
    #     for i in range(self.f_num):
    #         input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #         emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
    #         emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
    #         emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
    #         att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
    #         att_weights = tf.nn.softmax(tf.reshape(att, (self.y_num, -1, input_shape[1]**2)))  # (40, N, size**2)
    #         # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
    #         att_w_soft = tf.reshape(tf.transpose(att_weights, (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
    #         att_w.append(tf.expand_dims(att_w_soft,0))
    #     att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
    #     # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
    #     return att_w

    def __call__(self, inputs, labels, embs, emb_matrix, bs, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            input_shape = out.shape.as_list()
            # out = tf.concat([inputs, labels], 1)
            w = 4
            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            # [embs1, embs2, embs3, embs4] = embs
            noise_inputs = []
            for i in range(4):
                size = 2**i*4
                shape = [bs, size, size, 1]
                noise_input = tf.get_variable('noise%d'%i, shape=shape, initializer=tf.initializers.random_normal(),
                                              trainable=False)
                noise_inputs.append(noise_input)
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            # hidden = out
            # out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]
            #
            # # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            # #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            # emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            # emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            # weights = tf.matmul(out, emb_y)  # [N,4,40]
            # weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            # weight = tf.nn.softmax(weight)
            # weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            # hout = out * weight  # [N,4,z_dim]
            # out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            # out = tf.nn.relu(self.use_noise(out, noise_inputs[0],0))
            out = self.use_noise(out, noise_inputs[0],0)
            # out = apply_noise(out, noise_var=None, name='noise4', random_noise=True)
            # out = lrelu(apply_bias(out, name='n_bias4'))
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            with tf.name_scope("att4") as scope:
                # out = self.do_att(out, embs4)
                att4 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att4, is_training=is_training)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 16*16*256
            # out = tf.nn.relu(self.use_noise(out, noise_inputs[1], 1))
            out = self.use_noise(out, noise_inputs[1], 1)
            # out = apply_noise(out, noise_var=None, name='noise3', random_noise=True)
            # out = lrelu(apply_bias(out, name='n_bias3'))
            with tf.name_scope("att3") as scope:
                # out = self.do_att(out, embs3)
                att3 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att3, is_training=is_training)

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 32*32*128
            # out = tf.nn.relu(self.use_noise(out, noise_inputs[2], 2))
            out = self.use_noise(out, noise_inputs[2], 2)
            # out = apply_noise(out, noise_var=None, name='noise2', random_noise=True)
            # out = lrelu(apply_bias(out, name='n_bias2'))
            with tf.name_scope("att2") as scope:
                # out = self.do_att(out, embs2)
                att2 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att2, is_training=is_training)

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 64*64*64
            out = self.use_noise(out, noise_inputs[3], 3)
            # out = tf.nn.relu(self.use_noise(out, noise_inputs[3], 3))
            # out = apply_noise(out, noise_var=None, name='noise1', random_noise=True)
            # out = lrelu(apply_bias(out, name='n_bias1'))
            with tf.name_scope("att1") as scope:
                # out = self.do_att(out, embs1)
                att1 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att1, is_training=is_training)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Encoder128_style(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2,
                 f_num=4):
        self.name = 'Encoder128_style'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_wscale=self.use_wscale)
            # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
            #                          normalizer_params={'scale': True, 'is_training': training})
            # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

            style_xs = []
            for i in range(self.y_num):
                style_x = style_mod(x, latent_in[:,i,:], use_bias=True, name='Style%d'%i)   # latent_in(N, 40, c)
                style_xs.append(style_x)
            style_xs = tf.concat(style_xs, axis=3)
            out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                             normalizer_params={'scale': True, 'is_training': training})
            return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # noise_inputs = []
            # for i in range(4):
            #     size = 2**(3-i)*4
            #     shape = [bs, size, size, 1]
            #     noise_input = tf.get_variable('noise%d'%i, shape=shape, initializer=tf.initializers.random_normal(),
            #                                   trainable=False)
            #     noise_inputs.append(noise_input)

            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.layer_adain(out, embs, training=is_training, name='style1')

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.layer_adain(out, embs, training=is_training, name='style2')

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.layer_adain(out, embs, training=is_training, name='style3')

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            out = self.layer_adain(out, embs, training=is_training, name='style4')

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})


            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)



            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128_style(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2,
                 f_num=4):
        self.name = 'Generator128_style'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_wscale=self.use_wscale)
            # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
            #                          normalizer_params={'scale': True, 'is_training': training})
            # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

            style_xs = []
            for i in range(self.y_num):
                style_x = style_mod(x, latent_in[:,i,:], use_bias=True, name='Style%d'%i)   # latent_in(N, 40, c)
                style_xs.append(style_x)
            style_xs = tf.concat(style_xs, axis=3)
            out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                             normalizer_params={'scale': True, 'is_training': training})
            return out


    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            input_shape = out.shape.as_list()
            # out = tf.concat([inputs, labels], 1)
            w = 4
            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            # [embs1, embs2, embs3, embs4] = embs
            # noise_inputs = []
            # for i in range(4):
            #     size = 2**i*4
            #     shape = [bs, size, size, 1]
            #     noise_input = tf.get_variable('noise%d'%i, shape=shape, initializer=tf.initializers.random_normal(),
            #                                   trainable=False)
            #     noise_inputs.append(noise_input)
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            # hidden = out
            # out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]
            #
            # # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            # #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            # emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            # emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            # weights = tf.matmul(out, emb_y)  # [N,4,40]
            # weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            # weight = tf.nn.softmax(weight)
            # weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            # hout = out * weight  # [N,4,z_dim]
            # out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            out = self.layer_adain(out, embs, training=is_training, name='style4')


            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 16*16*256
            out = self.layer_adain(out, embs, training=is_training, name='style3')


            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 32*32*128
            out = self.layer_adain(out, embs, training=is_training, name='style2')


            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 64*64*64
            out = self.layer_adain(out, embs, training=is_training, name='style1')


            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Encoder128_avg(object):
    def __init__(self,
                 z_dim=256,
                 y_num=38,
                 y_dim=2,
                 f_num=4):
        self.name = 'Encoder128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
        #                          normalizer_params=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        w_outs = []
        att_w = []
        for i in range(self.y_num):#(f_num, N, 40, c)
            emb = embs[:, :, i, :]  # (f_num, N, c)
            emb = tf.reshape(emb, (self.f_num, -1, emb.shape.as_list()[-1], 1))  # (f_num, N, c, 1)
            input_r_ = tf.tile(tf.expand_dims(input_r,0), (self.f_num,1,1,1))  # (f_num, N, size**2, c)
            att = tf.matmul(input_r_, emb)  # (f_num, N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (self.f_num, -1, input_shape[1] ** 2)))  # (f_num, N, size**2)
            att_weight = tf.reduce_mean(att_weight, axis=0)   #(N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
            att_w.append(att_weight)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        att_w = tf.concat(att_w, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
        #                  normalizer_params=None)
        out = out + inputs
        return out, att_w

    # def do_att1(self, inputs, ksize, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #     input_r = tf.extract_image_patches(down_inputs, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME') #[N,w,h,k**2**c]
    #     input_r_shape = input_r.shape.as_list()
    #
    #     w_outs = []
    #     for i in range(self.y_num):  # # [40,N,32*4]
    #         emb = embs[i, :, :]  # (N, c*k*k)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c*k*k, 1)
    #         input_r_ = tf.reshape(input_r, (-1,input_r_shape[1] * input_r_shape[2], input_r_shape[3]))  # (N, size**2, k**2**c)
    #         att = tf.matmul(input_r_, emb)  # (N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out
    #
    # def do_att2(self, inputs, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(self.y_num):
    #         emb = embs[i, :, :]  # (N, c)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
    #         att = tf.matmul(input_r, emb)  # (N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att1") as scope:
                out = self.do_att(out, embs1)

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att2") as scope:
                out = self.do_att(out, embs2)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att3") as scope:
                out = self.do_att(out, embs3)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att4") as scope:
                out = self.do_att(out, embs4)

            # with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            #            out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            hidden = out
            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]

            # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            y_reshape = tf.transpose(labels, (1, 0, 2))  # [40,N,2]
            emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            weights = tf.matmul(out, emb_y)  # [N,4,40]
            weight = tf.reduce_mean(weights, axis=2, keep_dims=False)  # [N,4]
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            hout = out * weight  # [N,4,z_dim]
            out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Generator128_avg(object):
    def __init__(self,
                 z_dim=256,
                 y_num=38,
                 y_dim=2,
                 f_num=4):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_params=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(self.y_num):#(f_num, N, 40, c)
    #         emb = embs[:, :, i, :]  # (f_num, N, c)
    #         emb = tf.reshape(emb, (self.f_num, -1, emb.shape.as_list()[-1], 1))  # (f_num, N, c, 1)
    #         input_r_ = tf.tile(tf.expand_dims(input_r,0), (self.f_num,1,1,1))  # (f_num, N, size**2, c)
    #         att = tf.matmul(input_r_, emb)  # (f_num, N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (self.f_num, -1, input_shape[1] ** 2)))  # (f_num, N, size**2)
    #         att_weight = tf.reduce_mean(att_weight, axis=0)   #(N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_params=None)
    #     out = out + inputs
    #     return out
    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
        #                          normalizer_params=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

        w_outs = []
        att_w = []
        for i in range(self.y_num):#(f_num, N, 40, c)
            emb = embs[:, :, i, :]  # (f_num, N, c)
            emb = tf.reshape(emb, (self.f_num, -1, emb.shape.as_list()[-1], 1))  # (f_num, N, c, 1)
            input_r_ = tf.tile(tf.expand_dims(input_r,0), (self.f_num,1,1,1))  # (f_num, N, size**2, c)
            att = tf.matmul(input_r_, emb)  # (f_num, N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (self.f_num, -1, input_shape[1] ** 2)))  # (f_num, N, size**2)
            att_weight = tf.reshape(tf.transpose(att_weight,(1,2,0)), (-1, input_shape[1], input_shape[2], self.f_num, 1))  # (N, size, size, f_num, 1)
            att_w.append(att_weight)
        att_w = tf.concat(att_w, axis=4) # (N, size, size, f_num, 40)
        return att_w

    def do_adain(self, inputs, atts, is_training=True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)  #[N, 1, 1, c]
        adains = []

        for i in range(self.y_num):
            att = atts[:, :, :, :, i]  #[N, H, W, 4]
            mu_y, sigma_y = tf.nn.moments(att, axes=[1, 2], keep_dims=True) #[N, 1, 1, 4]
            group_adains = []
            g_len = input_shape[-1]/self.f_num
            for k in range(self.f_num):
                group_adain = tf.expand_dims(sigma_y[:,:,:,k],3) * x_[:,:,:,k*g_len:(k+1)*g_len] + tf.expand_dims(mu_y[:,:,:,k],3)
                group_adains.append(group_adain)
            adain = tf.concat(group_adains, axis=3)
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tf.reduce_sum(adains, axis=0)
        return out

    # def do_att1(self, inputs, ksize, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #     input_r = tf.extract_image_patches(down_inputs, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME') #[N,w,h,k**2**c]
    #     input_r_shape = input_r.shape.as_list()
    #
    #     w_outs = []
    #     for i in range(self.y_num):  # # [40,N,32*9]
    #         emb = embs[i, :, :]  # (N, c*k*k)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c*k*k, 1)
    #         input_r_ = tf.reshape(input_r, (-1,input_r_shape[1] * input_r_shape[2], input_r_shape[3]))  # (N, size**2, k**2**c)
    #         att = tf.matmul(input_r_, emb)  # (N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out
    #
    # def do_att2(self, inputs, embs, is_training=True):
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
    #     #                          normalizer_fn=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(self.y_num):  # [40,N,c]
    #         emb = embs[i, :, :]  # (N, c)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
    #         att = tf.matmul(input_r, emb)  # (N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None, normalizer_fn=tcl.batch_norm,
    #                                       normalizer_params={'scale': True, 'is_training': is_training})
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #     #                  normalizer_fn=None)
    #     out = out + inputs
    #     return out

    def __call__(self, inputs, labels, embs, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            # out = tf.concat([inputs, labels], 1)
            w = 4
            #            [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            # [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            # hidden = out
            # out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]
            #
            # # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            # #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            # y_reshape = tf.transpose(labels, (1, 0, 2))  # [40,N,2]
            # emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            # emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            # weights = tf.matmul(out, emb_y)  # [N,4,40]
            # weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            # weight = tf.nn.softmax(weight)
            # weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            # hout = out * weight  # [N,4,z_dim]
            # out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            with tf.name_scope("att4") as scope:
                att4 = self.do_att(out, embs, is_training=is_training)
            out = self.do_adain(out, att4, is_training=is_training)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope("att3") as scope:
                att3 = self.do_att(out, embs)
            out = self.do_adain(out, att3, is_training=is_training)

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope("att2") as scope:
                att2 = self.do_att(out, embs)
            out = self.do_adain(out, att2, is_training=is_training)

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope("att1") as scope:
                att1 = self.do_att(out, embs)
            out = self.do_adain(out, att1, is_training=is_training)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Generator128_avg_new(object):
    def __init__(self,
                 z_dim=256,
                 y_num=38,
                 y_dim=2,
                 f_num=4):
        self.name = 'Generator128'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num

    # def do_att(self, inputs, embs, is_training=True):
    #
    #     # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
    #                              normalizer_params=None)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(self.y_num):#(f_num, N, 40, c)
    #         emb = embs[:, :, i, :]  # (f_num, N, c)
    #         emb = tf.reshape(emb, (self.f_num, -1, emb.shape.as_list()[-1], 1))  # (f_num, N, c, 1)
    #         input_r_ = tf.tile(tf.expand_dims(input_r,0), (self.f_num,1,1,1))  # (f_num, N, size**2, c)
    #         att = tf.matmul(input_r_, emb)  # (f_num, N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (self.f_num, -1, input_shape[1] ** 2)))  # (f_num, N, size**2)
    #         att_weight = tf.reduce_mean(att_weight, axis=0)   #(N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tcl.batch_norm,
    #     #                                   normalizer_params={'scale': True, 'is_training': is_training})
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #                      normalizer_params=None)
    #     out = out + inputs
    #     return out

    def do_att1(self, inputs, ksize, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # down_inputs = tcl.conv2d(inputs, num_outputs=16, kernel_size=1, stride=1, activation_fn=None,
        #                          normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)
        input_r = tf.extract_image_patches(down_inputs, [1,ksize,ksize,1], [1,1,1,1], [1,1,1,1], padding='SAME') #[N,w,h,k**2**c]
        input_r_shape = input_r.shape.as_list()

        w_outs = []
        for i in range(self.y_num):  # # [40,N,32*9]
            emb = embs[i, :, :]  # (N, c*k*k)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c*k*k, 1)
            input_r_ = tf.reshape(input_r, (-1,input_r_shape[1] * input_r_shape[2], input_r_shape[3]))  # (N, size**2, k**2**c)
            att = tf.matmul(input_r_, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)

        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
        #                  normalizer_fn=None)
        out = out + inputs
        return out

    def do_att2(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
        #                          normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)

        w_outs = []
        for i in range(self.y_num):  # [40,N,c]
            emb = embs[i, :, :]  # (N, c)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        # out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
        #                  normalizer_fn=None)
        out = out + inputs
        return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            # out = tf.concat([inputs, labels], 1)
            w = 4
            #            [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4] = embs
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            hidden = out
            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]

            # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            # y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            y_reshape = tf.transpose(labels, (1, 0, 2))  # [40,N,2]
            emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            weights = tf.matmul(out, emb_y)  # [N,4,40]
            weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            hout = out * weight  # [N,4,z_dim]
            out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            with tf.name_scope("att4") as scope:
                out = self.do_att2(out, embs1)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (8, 8))  # 8*8*128
            with tf.name_scope("att3") as scope:
                out = self.do_att1(out, 3, embs2)

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (16, 16))  # 16*16*64
            with tf.name_scope("att2") as scope:
                out = self.do_att1(out, 3, embs3)

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (32, 32))  # 32*32*32
            with tf.name_scope("att1") as scope:
                out = self.do_att1(out, 3, embs4)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.image.resize_images(out, (64, 64))
            with tf.name_scope('conv1') as scope:
                out = tcl.conv2d(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Encoder64(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2):
        self.name = 'Encoder64'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
                                 normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)

        w_outs = []
        for i in range(self.y_num):
            emb = embs[:, i, :]  # (N, c)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
                         normalizer_fn=None)
        out = out + inputs
        return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4, embs5] = embs
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(inputs, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att1") as scope:
                out = self.do_att(out, embs1)

            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att2") as scope:
                out = self.do_att(out, embs2)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att3") as scope:
                out = self.do_att(out, embs3)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)
            with tf.name_scope("att4") as scope:
                out = self.do_att(out, embs4)

            with tf.name_scope('conv5_1') as scope:
               out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=tcl.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv5_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=tcl.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv5_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                    normalizer_fn=tcl.batch_norm,
                                    normalizer_params={'scale': True, 'is_training': is_training})
            # out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)
            with tf.name_scope("att5") as scope:
                out = self.do_att(out, embs5)

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            # hidden = out
            #            embs_matrix = tf.nn.embedding_lookup(embs_matrix, labels)
            #            embs_matrix = tf.reshape(embs_matrix, (-1, self.z_dim, 1))
            #            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1]/self.z_dim, self.z_dim))
            #            weight = tf.matmul(out, embs_matrix)
            #            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1]/self.z_dim))
            #            weight = tf.nn.softmax(weight)
            #            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1]/self.z_dim, 1))
            #            out = out * weight
            #            out = tf.reduce_sum(out, axis = 1)
            hidden = out
            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]

            # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            weights = tf.matmul(out, emb_y)  # [N,4,40]
            weight = tf.reduce_mean(weights, axis=2, keep_dims=False)  # [N,4]
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            hout = out * weight  # [N,4,z_dim]
            out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator64(object):
    def __init__(self,
                 z_dim=256,
                 y_num=40,
                 y_dim=2):
        self.name = 'Generator64'
        self.z_dim = z_dim
        self.y_num = y_num
        self.y_dim = y_dim

    def do_att(self, inputs, embs, is_training=True):

        down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=None,
                                 normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 32))  # (N, size**2, c)

        w_outs = []
        for i in range(self.y_num):
            emb = embs[:, i, :]  # (N, c)
            emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
            att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            w_out = tf.multiply(down_inputs, att_weight)  # (N, size, size, c)
            w_outs.append(w_out)
        w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
        out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
                         normalizer_fn=None)
        out = out + inputs
        return out

    # def do_att(self, inputs, embs, is_training=True):
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
    #
    #     w_outs = []
    #     for i in range(40):
    #         emb = embs[:, i, :]  # (N, c)
    #         emb = tf.reshape(emb, (-1, emb.shape.as_list()[-1], 1))  # (N, c, 1)
    #         att = tf.matmul(input_r, emb)  # (N, size**2, 1)
    #         # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
    #         att_weight = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))  # (N, size**2)
    #         att_weight = tf.reshape(att_weight, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         w_out = tf.multiply(inputs, att_weight)  # (N, size, size, c)
    #         w_outs.append(w_out)
    #     w_outs = tf.concat(w_outs, axis=3)  # (N, size, size, 40*c)
    #     out = tcl.conv2d(w_outs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=None,
    #                      normalizer_fn=None)
    #     out = out + inputs
    #     return out

    def __call__(self, inputs, labels, embs, emb_matrix, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            # out = tf.concat([inputs, labels], 1)
            w = 4
            #            [embs1, embs2, embs3, embs4, embs5, embs6, embs7, embs8] = embs
            [embs1, embs2, embs3, embs4, embs5] = embs
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
                # with tf.name_scope('fc7') as scope:
                #     out = tcl.fully_connected(out, 4096, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                #                               normalizer_params={'scale': True, 'is_training': is_training})

            #            embs_matrix = tf.nn.embedding_lookup(embs_matrix, labels)
            #            embs_matrix = tf.reshape(embs_matrix, (-1, self.z_dim, 1))
            #            out = tf.reshape(out, (-1, 1024/self.z_dim, self.z_dim))
            #            weight = tf.matmul(out, embs_matrix)
            #            weight = tf.reshape(weight, (-1, 1024/self.z_dim))
            #            weight = tf.nn.softmax(weight)
            #            weight = tf.reshape(weight, (-1, 1024/self.z_dim, 1))
            #            out = out * weight
            #            out = tf.reduce_sum(out, axis = 1)
            hidden = out
            out = tf.reshape(out, (-1, hidden.shape.as_list()[-1] / self.z_dim, self.z_dim))  # [N,4,z_dim]

            # emb_matrix = tf.get_variable("emb4", [self.y_num, self.y_dim, self.z_dim], dtype=tf.float32,
            #                          initializer=tcl.xavier_initializer())  # [40,2,z_dim]
            y_reshape = tf.reshape(labels, (self.y_num, -1, self.y_dim))  # [40,N,2]
            emb_y = tf.transpose(tf.matmul(y_reshape, emb_matrix), [1, 2, 0])  # [40,N,z_dim]->[N,z_dim,40]
            emb_y = tcl.batch_norm(emb_y, activation_fn=tf.nn.relu, scale=True)
            weights = tf.matmul(out, emb_y)  # [N,4,40]
            weight = tf.reduce_mean(weights, axis=2)  # [N,4]
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (-1, hidden.shape.as_list()[-1] / self.z_dim, 1))  # [N,4,1]
            hout = out * weight  # [N,4,z_dim]
            out = tf.reduce_sum(hout, axis=1)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512
            with tf.name_scope("att5") as scope:
                out = self.do_att(out, embs5)

            with tf.name_scope('conv5_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                normalizer_fn=tcl.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv5_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=tcl.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv5_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                normalizer_fn=tcl.batch_norm,
                                normalizer_params={'scale': True, 'is_training': is_training})  #8*8*512
            with tf.name_scope("att4") as scope:
                out = self.do_att(out, embs4)

            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 16*16*256
            with tf.name_scope("att3") as scope:
                out = self.do_att(out, embs3)

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 32*32*128
            with tf.name_scope("att2") as scope:
                out = self.do_att(out, embs2)

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 64*64*64
            with tf.name_scope("att1") as scope:
                out = self.do_att(out, embs1)

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Discriminator128(object):
    def __init__(self):
        self.name = 'Discriminator128'
        self.channel = 32

    def __call__(self, inputs, labels, is_training = True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = lrelu(conv(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn = True, scope='conv_11'))
            out = lrelu(conv(out, channels=self.channel*2, kernel=3, stride=2, pad=1, sn = True, scope='conv_12'))
            
            out = lrelu(conv(out, channels=self.channel*4, kernel=3, stride=1, pad=1, sn = True, scope='conv_21'))
            out = lrelu(conv(out, channels=self.channel*4, kernel=3, stride=2, pad=1, sn = True, scope='conv_22'))
            
            out = lrelu(conv(out, channels=self.channel*8, kernel=3, stride=1, pad=1, sn = True, scope='conv_31'))
            
            out = lrelu(conv(out, channels=self.channel*8, kernel=3, stride=2, pad=1, sn = True, scope='conv_41'))
            out = lrelu(conv(out, channels=self.channel*8, kernel=3, stride=2, pad=1, sn = True, scope='conv_42'))
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            #
            # out = tcl.flatten(out)
            out = tf.reduce_sum(out, axis=(1, 2))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')

            # emb_y = fully_connected(labels, self.channel * 8, sn=True, scope='y_proj')
            # proj = tf.reduce_sum(emb_y * out, axis=1, keep_dims=True)
            # y_onehot = tf.one_hot(y, num_classes)
            # y_proj = fully_connected(y_onehot, 1024, sn = true, scope='y_proj')
            # y_proj = tf.reduce_sum(y_proj * out, axis=1, keep_dims=True)
            

            pros = []
            for i in range(40):
                w_y = labels[:,i,:]
                emb_y = fully_connected(w_y, self.channel * 8, use_bias=False, sn=True, scope='y_proj'+str(i))
                pro = tf.reduce_sum(emb_y * out, axis=1, keepdims=True)
                pros.append(pro)
            pros = tf.concat(pros, axis=1)
            y_proj = tf.reduce_sum(pros, axis=1, keepdims=True)
            
#            q = tcl.fully_connected(out, 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm,
#                                      normalizer_params={'scale': True, 'is_training': is_training})
#
#            q = tcl.fully_connected(q, self.z_dim, activation_fn=None)

            return d+y_proj*0.001

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Discriminator256_RC(object):
    def __init__(self, y_dim=20, y_num=1):
        self.name = 'Discriminator256_RC'
        self.channel = 32
        self.y_dim = y_dim
        self.y_num = y_num

    def __call__(self, inputs, pa=False, pa_again=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))#128
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))#64
            
            out_c = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21_c'))
            out_c = lrelu(conv_2d(out_c, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22_c'))#32 
            out_c = lrelu(conv_2d(out_c, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31_c')) 
            out_c = lrelu(conv_2d(out_c, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41_c'))#16
            out_c = conv_2d(out_c, channels=self.y_dim, kernel=1, stride=1, pad=0, sn=True, name='conv_42_c')
            
            
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))#32

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))#16
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))#8
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))#4
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))#2
                
            out = tf.reduce_sum(out, axis=(1, 2))
            
            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')
#            d = sigmoid(fully_connected(out, 1, use_bias=False, sn=True, scope='o_1'))

            return d, out_c

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class CDiscriminator256(object):
    def __init__(self, y_dim=20, y_num=1):
        self.name = 'Discriminator256_RC'
        self.channel = 32
        self.y_dim = y_dim
        self.y_num = y_num

    def __call__(self, inputs, condition, pa=False, pa_again=False, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            inputs = tf.concat([inputs, condition], -1)
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))#128
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))#64
            
#            out_c = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21_c'))
#            out_c = lrelu(conv_2d(out_c, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22_c'))#32 
#            out_c = lrelu(conv_2d(out_c, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31_c')) 
#            out_c = lrelu(conv_2d(out_c, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41_c'))#16
#            out_c = conv_2d(out_c, channels=self.y_dim, kernel=1, stride=1, pad=0, sn=True, name='conv_42_c')
            
            
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))#32

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))#16
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))#8
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))#4
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))#2
                
            out = tf.reduce_sum(out, axis=(1, 2))
            
            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')
#            d = sigmoid(fully_connected(out, 1, use_bias=False, sn=True, scope='o_1'))

            return d

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Discriminatorall_AC(object):
    def __init__(self, y_dim=2, y_num=40):
        self.name = 'Discriminatorall_AC'
        self.channel = 32
        self.y_dim = y_dim
        self.y_num = y_num

    def __call__(self, inputs, pa=False, pa_again=False, is_training=True, reuse=False, use_bias=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
#            label = inputs[1]
#            inputs = inputs[0]
#            input_shape = inputs.shape.as_list()
#            label = tf.tile(tf.expand_dims(tf.expand_dims(label, 1),1), [1, input_shape[1], input_shape[2], 1])
#            all_inputs = tf.concat([inputs, label], -1)
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))

            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

#            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))
#            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            #
            # out = tcl.flatten(out)
#            out = tf.reduce_sum(out, axis=(1, 2))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))
            d0 = conv_2d(out, channels = self.channel * 8, kernel=4, stride=4, pad=0, sn=True, name='o_10')
            d00 = conv_2d(d0, channels = 1, kernel=1, stride=1, pad=0, sn=True, name='o_100')

            # cls=[]
            # for i in range(self.y_num):
            #     clsi = fully_connected(out, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #     cls.append(clsi)
            # y_cls = tf.reshape(tf.concat(cls, axis=1),(-1, self.y_num, self.y_dim))
            y_cls0 = conv_2d(out, channels = self.channel * 8, kernel=4, stride=4, pad=0, sn=True, name='cls_o10')
            y_cls00 = conv_2d(y_cls0, channels = self.y_dim, kernel=1, stride=1, pad=0, sn=True, name='cls_o100')

            return d00, y_cls00

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class DRGAN_Discriminator(object):
    def __init__(self, y_dim=2, ID_dim=2, y_num=40):
        self.name = 'DRGAN_Discriminator'
        self.channel = 32
        self.y_dim = y_dim
        self.ID_dim = ID_dim
        self.y_num = y_num

    def __call__(self, inputs, pa=False, pa_again=False, is_training=True, reuse=False, use_bias=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))

            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            #
            # out = tcl.flatten(out)
            out = tf.reduce_sum(out, axis=(1, 2))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')

            # cls=[]
            # for i in range(self.y_num):
            #     clsi = fully_connected(out, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #     cls.append(clsi)
            # y_cls = tf.reshape(tf.concat(cls, axis=1),(-1, self.y_num, self.y_dim))
            y_cls = fully_connected(out, self.y_dim, use_bias=use_bias, sn=True, scope='cls')
            y_cls_id = fully_connected(out, self.ID_dim, use_bias=use_bias, sn=True, scope='cls_id')

            return d, y_cls, y_cls_id

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
class Discriminator128_AC(object):
    def __init__(self, y_dim=2, y_num=40):
        self.name = 'Discriminator128_AC'
        self.channel = 32
        self.y_dim = y_dim
        self.y_num = y_num

    def __call__(self, inputs, pa=False, pa_again=False, is_training=True, reuse=False, use_bias=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))

            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))
            # out = tcl.avg_pool2d(out, kernel_size=2, stride=2)
            #
            # out = tcl.flatten(out)
            out = tf.reduce_sum(out, axis=(1, 2))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            d = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')

            # cls=[]
            # for i in range(self.y_num):
            #     clsi = fully_connected(out, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #     cls.append(clsi)
            # y_cls = tf.reshape(tf.concat(cls, axis=1),(-1, self.y_num, self.y_dim))
            y_cls = fully_connected(out, self.y_dim, use_bias=use_bias, sn=True, scope='cls')

            return d, y_cls

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            
class Discriminator128_mhD(object):
    def __init__(self, y_dim=2, y_num=40):
        self.name = 'Discriminator128_mhD'
        self.channel = 32
        self.y_dim = y_dim
        self.y_num = y_num

    def __call__(self, inputs, label, pa=False, pa_again=False, is_training=True, reuse=False, use_bias=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out = lrelu(conv_2d(inputs, channels=self.channel, kernel=3, stride=2, pad=1, sn=True, name='conv_11'))
            out = lrelu(conv_2d(out, channels=self.channel * 2, kernel=3, stride=2, pad=1, sn=True, name='conv_12'))

            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=1, pad=1, sn=True, name='conv_21'))
            out = lrelu(conv_2d(out, channels=self.channel * 4, kernel=3, stride=2, pad=1, sn=True, name='conv_22'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=1, pad=1, sn=True, name='conv_31'))

            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_41'))
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_42'))
            if pa:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_43'))
            if pa_again:
                out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=3, stride=2, pad=1, sn=True, name='conv_44'))
            out = lrelu(conv_2d(out, channels=self.channel * 8, kernel=4, stride=1, pad=0, sn=True, name='conv_5'))
            out = lrelu(conv_2d(out, channels=self.y_dim, kernel=1, stride=1, pad=0, sn=True, name='conv_6'))
            out = tcl.flatten(out)
            n_index = tf.expand_dims(tf.constant(range(out.shape[0])), -1)
            y_index = tf.expand_dims(label, -1)
            index = tf.concat([n_index, y_index], -1)
            out = tf.gather_nd(out, index)
            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Encoder128_cvae(object):
    def __init__(self, z_dim=256, y_num=40):
        self.name = 'Encoder128_cvae'
        self.z_dim = z_dim
        self.y_num = y_num


    def __call__(self, inputs, labels, is_training=True, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            out = inputs
            input_shape = inputs.shape.as_list()
            for i in range(self.y_num):
                label = labels[:, i, :]    #[N, 2]
                y = tf.reshape(label, (-1, 1, 1, 2))
                y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
                out = tf.concat([out, y], 3)

            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)


            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                 normalizer_fn=tcl.batch_norm,
                                 normalizer_params={'scale': True, 'is_training': is_training})
            out = tcl.avg_pool2d(out, kernel_size=2, stride=2)


            # with tf.name_scope('conv5_1') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_2') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            with tf.name_scope('conv5_3') as scope:
            #                out = tcl.conv2d(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
            #                                 normalizer_fn=tcl.batch_norm,
            #                                 normalizer_params={'scale': True, 'is_training': is_training})
            #            out = tcl.max_pool2d(out, kernel_size=2, stride=2)

            #            out = tcl.avg_pool2d(out, kernel_size= 2, stride=2)
            #            with tf.name_scope("att5") as scope:
            #                out = self.do_att(out, embs5)

            out = tcl.flatten(out)

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})


            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Generator128_cvae(object):
    def __init__(self, z_dim=256):
        self.name = 'Generator128_cvae'
        self.z_dim = z_dim

    def __call__(self, inputs, labels, is_training=True, reuse=False):
        # 2 fully-connected layers, followed by 6 deconv layers with 2-by-2 upsampling
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # z = tf.concat([z_c, z_p], axis = 1)
            out = inputs
            out = tf.concat([inputs, labels], 1)
            w = 4
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(out, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * 256, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            out = tf.reshape(out, (-1, w, w, 256))  # 4*4*512


            with tf.name_scope('conv4_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv4_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 16*16*256

            with tf.name_scope('conv3_3') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=128, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv3_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 32*32*128

            with tf.name_scope('conv2_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv2_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})  # 64*64*64

            with tf.name_scope('conv1_2') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=32, kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                           normalizer_fn=tcl.batch_norm,
                                           normalizer_params={'scale': True, 'is_training': is_training})
            with tf.name_scope('conv1_1') as scope:
                out = tcl.conv2d_transpose(out, num_outputs=3, kernel_size=3, stride=1, activation_fn=tf.nn.tanh)

            return out

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class styleganclassifier(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def __call__(self, conv, batch_size, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse == True:
                scope.reuse_variables()
            conv = fully_connected(conv, 1024, use_bias=True, sn=False, scope='cls_fc_1')
            conv = fully_connected(conv, 1024, use_bias=True, sn=False, scope='cls_fc_2')

            y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=False, scope='cls')

            return y_cls

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegandiscriminate_concat(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(256 / (2 **(stage * 1)), 128)

    def __call__(self, conv, batch_size, reuse=False, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name) as scope:

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # fromRGB
            input_shape = conv.shape.as_list()
            for i in range(self.y_num):
                label = labels[:, i, :]  # [N, 2]
                y = tf.reshape(label, (-1, 1, 1, 2))
                y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
                conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            # return tf.nn.sigmoid(output), output
            return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class stylegandiscriminate_noc_ac(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')

            hidden = conv
#            emb_y = fully_connected(labels, conv.shape[-1], use_bias=False, sn=True, scope='y_proj')
#            proj = tf.reduce_sum(emb_y * hidden, axis=1, keep_dims=True)
            y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')
            # if self.y_num > 1:
            #     cls = []
            #     for i in range(self.y_num):
            #         # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
            #         clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #         cls.append(clsi)
            #     y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            # else:
            #     y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')


            return output, y_cls
#            return output + proj
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
    
class Dstylegandiscriminate_noc_ac(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')

            hidden = conv
#            emb_y = fully_connected(labels, conv.shape[-1], use_bias=False, sn=True, scope='y_proj')
#            proj = tf.reduce_sum(emb_y * hidden, axis=1, keep_dims=True)
            y_cls = fully_connected(conv, 512, use_bias=False, sn=True, scope='cls')
            y_cls = fully_connected(y_cls, 256, use_bias=False, sn=True, scope='cls1')
            y_cls = fully_connected(y_cls, self.y_dim, use_bias=False, sn=True, scope='cls2')
            # if self.y_num > 1:
            #     cls = []
            #     for i in range(self.y_num):
            #         # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
            #         clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #         cls.append(clsi)
            #     y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            # else:
            #     y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')


            return output, y_cls
#            return output + proj
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegandiscriminate_noc_feature(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            feature_loss = []
            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, \
                                sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))
            feature_loss.append(conv)
            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden
                feature_loss.append(conv)
                
            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])
            feature_loss.append(conv)
            
            #for D
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
            return output + proj, feature_loss
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class stylegandiscriminate_noc(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, \
                                sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
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

class stylegandiscriminate_noc_pool(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, \
                                sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reduce_mean(conv, axis=[1, 2])
#            conv = tf.reshape(conv, [batch_size, -1])

            #for D
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
    
class stylegandiscriminate_noc_clsID(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, yid_dim=250 ):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim 
        self.yid_dim = yid_dim

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, labels=None, idlabels=None, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            # if reuse == True:
            #     scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB

            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, \
                                sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = lrelu(conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv
                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,
                #            name='dis_n_conv_3_{}'.format(conv.shape[1])))

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')
            
            y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')
            yid_cls = fully_connected(conv, self.yid_dim, use_bias=False, sn=True, scope='cls_id')
            
#            hidden = conv
#            emb_y = fully_connected(idlabels, conv.shape[-1], use_bias=False, sn=True, scope='yid_proj')
#            projid = tf.reduce_sum(emb_y * hidden, axis=1, keep_dims=True)
#
#            hidden = conv
#            emb_y = fully_connected(labels, conv.shape[-1], use_bias=False, sn=True, scope='y_proj')
#            proj = tf.reduce_sum(emb_y * hidden, axis=1, keep_dims=True)

            # if self.y_num > 1:
            #     cls = []
            #     for i in range(self.y_num):
            #         # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
            #         clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #         cls.append(clsi)
            #     y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            # else:
            #     y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')


            return output, y_cls, yid_cls
#            return output + proj, output + projid
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
class stylegandiscriminate(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=64):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel

    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)
        # return min(512 / (2 **(stage * 1)), 256)

    def __call__(self, conv, batch_size, reuse=tf.AUTO_REUSE, pg=5, t=False, alpha_trans=0.01):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))
            conv = lrelu(conv2d(conv, output_dim=64, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=True,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                if i<2:
                    c1 = c2 = 1
                else:
                    c1 = 2**(i-2)
                    c2 = c1*2
                #pg-2-i  pg-1-i  pg-2-i
                res = conv
                res = lrelu(conv2d(res, output_dim=self.channel*c2, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='res_{}'.format(res.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.channel*c1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.channel*c2, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=True,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=True, scope='o_1')
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            if self.y_num > 1:
                cls = []
                for i in range(self.y_num):
                    # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
                    clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
                    cls.append(clsi)
                y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            else:
                y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')

            # z_code = fully_connected(conv, zn_size, use_bias=False, sn=True, scope='z_code')
            # y_cls = fully_connected(conv, zn_size, use_bias=False, sn=True, scope='cls')

            # return tf.nn.sigmoid(output), output
            # return output, y_cls, z_code
            return output, y_cls

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegandiscriminate_AC(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegandiscriminate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        # return min(512 / (2 **(stage * 1)), 256)
        return min(1024 / (2 **(stage * 1)), 512)

    def layer_adain(self, x, latent_in, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_bias=True)
            out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs, batch_size, reuse=False, pg=5, t=False, alpha_trans=0.01):
        with tf.variable_scope(self.name) as scope:

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
            #                             initializer=tf.initializers.zeros(), trainable=False)

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_l_avg, embs, self.trunc_psi)

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = self.layer_adain(conv, embs, name='style1%d' % i)

                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = self.layer_adain(conv, embs, name='style2%d' % i)

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            output = fully_connected(conv, 1, use_bias=False, sn=False, scope='o_1')
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            # if self.y_num > 1:
            #     cls = []
            #     for i in range(self.y_num):
            #         # clsi = tcl.fully_connected(conv, self.y_dim, activation_fn=None, biases_initializer=None, scope='cls_'+str(i))
            #         clsi = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls_'+str(i))
            #         cls.append(clsi)
            #     y_cls = tf.reshape(tf.concat(cls, axis=1), (-1, self.y_num, self.y_dim))
            # else:
            #     y_cls = fully_connected(conv, self.y_dim, use_bias=False, sn=True, scope='cls')
            z_code = fully_connected(conv, self.z_dim, use_bias=False, sn=False, scope='z_code')
            # return tf.nn.sigmoid(output), output
            # return output, y_cls
            return output, z_code
            # return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    # def do_mynorm(self, inputs, atts, is_training = True):
    #     input_shape = inputs.shape.as_list()
    #     mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
    #     x_ = (inputs-mu_x)/(sigma_x+1e-6)
    #     adains = []
    #     for i in range(self.y_num):
    #         # att = tf.expand_dims(atts[:,:,:,i],3)
    #         # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
    #         mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
    #         sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
    #         # att = tf.reshape(atts[0, :, :, :, i], (-1, input_shape[1] ** 2))  # (N, size,size)
    #         # mu_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
    #         #                            normalizer_params={'scale': True, 'is_training': is_training})
    #         # sigma_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
    #         #                               normalizer_params={'scale': True, 'is_training': is_training})
    #         adain = sigma_y*x_+mu_y
    #         adain = tf.nn.leaky_relu(adain)
    #         adains.append(adain)
    #     adain_x = tf.concat(adains, axis=3)
    #     out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
    #                              # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
    #                              normalizer_fn=None)
    #     # out = tcl.instance_norm(out)
    #     # out = tf.reduce_mean(adains, axis=0)
    #     return out
    #
    # def do_att(self, inputs, embs, is_training=True):#multi
    #
    #     down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
    #                              # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
    #                              normalizer_fn=None)
    #     # down_inputs = tcl.instance_norm(down_inputs)
    #     input_shape = inputs.shape.as_list()
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)
    #
    #     att_w = []
    #     for i in range(self.f_num):
    #         input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
    #         emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
    #         emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
    #         emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
    #         att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
    #         # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
    #         att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
    #         att_w.append(tf.expand_dims(att_w_soft,0))
    #     att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
    #     # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
    #     return att_w

    def do_mynorm(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)

        mu_y = atts[0]
        sigma_y = atts[1]

        adain = sigma_y * x_ + mu_y
        # adain = (1+sigma_y) * x_ + mu_y
        return adain
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label

        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
        embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
        embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            # att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
        return atts
    def __call__(self, inputs, embs_l, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
            #                           initializer=tf.initializers.zeros(), trainable=False)

            emb_l_avg = tf.get_variable('embs_n_avg', shape=(1, self.channel), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)


            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))
                conv = de

                with tf.variable_scope('do_norm_1_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                    # de = lrelu(de)
                    att = self.do_att(de, embs_l, name='att_1_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)

                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                with tf.variable_scope('do_norm_2_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                    # de = lrelu(de)
                    att = self.do_att(de, embs_l, name='att_2_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_res(object):
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
        return min(512 / (2 **(stage * 1)), 256)
        # return min(1024 / (2 **(stage * 1)), 512)

    def do_mynorm(self, inputs, atts, is_training=True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)
        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0, :, :, :, i], 3)
            sigma_y = tf.expand_dims(atts[1, :, :, :, i], 3)
            # att = tf.reshape(atts[0, :, :, :, i], (-1, input_shape[1] ** 2))  # (N, size,size)
            # mu_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                            normalizer_params={'scale': True, 'is_training': is_training})
            # sigma_y = tcl.fully_connected(att, input_shape[3], activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            adain = sigma_y * x_ + mu_y
            adain = tf.nn.leaky_relu(adain)
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                         # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
                         normalizer_fn=None)
        # out = tcl.instance_norm(out)
        # out = tf.reduce_mean(adains, axis=0)
        return out

    def do_att(self, inputs, embs, name=None, is_training=True):  # multi

        down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
                                 normalizer_fn=None)
        # down_inputs = tcl.instance_norm(down_inputs)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num, 1, 1, 1))  # (40, N, size**2, c)
            emb = embs[i, :, :, :]  # (f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1, 0, 2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  # [40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft, 0))
        att_w = tf.concat(att_w, axis=0)  # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    # def do_mynorm(self, inputs, atts, name=None, is_training=False):  # only z, no label
    #     input_shape = inputs.shape.as_list()
    #     # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
    #     mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
    #     x_ = (inputs - mu_x) / (sigma_x + 1e-6)
    #
    #     mu_y = atts[0]
    #     sigma_y = atts[1]
    #
    #     adain = sigma_y * x_ + mu_y
    #     # adain = (1+sigma_y) * x_ + mu_y
    #     return adain
    # def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
    #
    #     input_shape = inputs.shape.as_list()
    #     down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
    #                              # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
    #                              normalizer_fn=None)
    #     input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)
    #
    #     embs = tf.reshape(embs_in, (-1, self.channel, self.f_num))  # (N, c, f_num)
    #
    #     # input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
    #     # embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
    #     # embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)
    #
    #     atts = []
    #     for i in range(self.f_num):
    #         emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
    #         att = tf.matmul(input_r, emb)  # (N, size**2, 1)
    #         att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
    #         atts.append(att_weight)
    #     # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
    #     return atts
    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         out = style_mod(x, latent_in, use_bias=True)
    #         return out
    def __call__(self, inputs, embs_l, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse) as scope:

            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            # de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
                                      initializer=tf.initializers.zeros(), trainable=False)
            # emb_l_avg = tf.get_variable('embs_n_avg', shape=(1, embs_l.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                # batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)

            de = tf.get_variable('const', shape=[1, 4, 4, self.get_nf(1)], initializer=tf.initializers.ones())
            de = tf.tile(tf.cast(de, tf.float32), [shape[0], 1, 1, 1])
            with tf.variable_scope('const_input_{}'.format(de.shape[1])):
                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_{}'.format(de.shape[1]))
                de = lrelu(apply_bias(de))
                att = self.do_att(de, embs_l, name='att_{}'.format(de.shape[1]), is_training=is_training)
                de = self.do_mynorm(de, att, is_training=is_training)


            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                if i<3:
                    c = i+1
                else:
                    c = 4
                de = upscale(de, 2)
                res = de
                res = lrelu(conv2d(res, output_dim=self.get_nf(c), d_w=1, d_h=1, use_wscale=self.use_wscale, name='res_{}'.format(res.shape[1])))
                with tf.variable_scope('res_norm_{}'.format(res.shape[1])):
                    res = apply_noise(res, noise_var=None, random_noise=True, name='noise_{}'.format(res.shape[1]))
                    res = lrelu(apply_bias(res))
                    att = self.do_att(res, embs_l, name='att_{}'.format(res.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    res = self.do_mynorm(res, att, is_training=is_training)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style1%d' % i)

                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(c), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))

                with tf.variable_scope('do_norm_1_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                    de = lrelu(apply_bias(de))
                    att = self.do_att(de, embs_l, name='att_1_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style1%d' % i)

                de = lrelu(conv2d(de, output_dim=self.get_nf(c), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                with tf.variable_scope('do_norm_2_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                    de = lrelu(apply_bias(de))
                    att = self.do_att(de, embs_l, name='att_2_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style2%d' % i)

                de = res + de

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_res1(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def do_mynorm(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)

        mu_y = atts[0]
        sigma_y = atts[1]

        adain = sigma_y * x_ + mu_y
        # adain = (1+sigma_y) * x_ + mu_y
        return adain
    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label

        input_shape = inputs.shape.as_list()
        down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
                                 normalizer_fn=None)
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        embs = tf.reshape(embs_in, (-1, self.channel, self.f_num))  # (N, c, f_num)

        # input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
        # embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
        # embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
        return atts
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            return out
    def __call__(self, inputs, embs_l, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            emb_l_avg = tf.get_variable('embs_n_avg', shape=(1, embs_l.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                res = de
                res = conv2d(res, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale,
                           name='res_{}'.format(res.shape[1]))
                with tf.variable_scope('res_norm_{}'.format(res.shape[1])):
                    res = apply_noise(res, noise_var=None, random_noise=True, name='noise_{}'.format(res.shape[1]))
                    res = lrelu(res)
                    att = self.do_att(res, embs_l, name='att_{}'.format(res.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    res = self.do_mynorm(res, att, is_training=is_training)
                    res = lrelu(res)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style1%d' % i)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))

                with tf.variable_scope('do_norm_1_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                    de = lrelu(de)
                    att = self.do_att(de, embs_l, name='att_1_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)
                    de = lrelu(de)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style1%d' % i)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))

                with tf.variable_scope('do_norm_2_{}'.format(de.shape[1])):
                    de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                    de = lrelu(de)
                    att = self.do_att(de, embs_l, name='att_2_{}'.format(de.shape[1]), is_training=is_training)
                    # att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                    de = self.do_mynorm(de, att, is_training=is_training)
                    de = lrelu(de)
                # de = self.layer_adain(de, embs_l, training=is_training, name='style2%d' % i)

                de = res + de

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_concat(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def __call__(self, inputs, pg=5, t=False, alpha_trans=0.0, labels=None, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            inputs = tf.concat((inputs, labels), 1)
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))


            # w = 7
            # with tf.name_scope('fc8') as scope:
            #     out = tcl.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            #
            # with tf.name_scope('fc6') as scope:
            #     out = tcl.fully_connected(out, w * w * int(self.get_nf(1)), activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            # de = tf.reshape(out, (-1, w, w, int(self.get_nf(1))))  # 4*4*512

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                res = de
                res = conv2d(res, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale,
                           name='res_{}'.format(res.shape[1]))
                de = batch_norm(de, scope='batch_norm_res_{}'.format(de.shape[1]))
                res = lrelu(res)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))
                de = batch_norm(de, scope='batch_norm_1_{}'.format(de.shape[1]))
                de = lrelu(de)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))
                de = batch_norm(de, scope='batch_norm_2_{}'.format(de.shape[1]))
                de = lrelu(de)

                de = res + de

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_id_mix(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def do_mynorm(self, inputs, atts, name=None, is_training = True):
        input_shape = inputs.shape.as_list()
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)

        # sw = tf.get_variable('l_multi_'+name, shape=(self.f_num/2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)

        adains = []
        for i in range(self.y_num):
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            # mu_y = tf.reduce_mean(tf.expand_dims(atts[0:5,:,:,:,i],4), axis=0, keep_dims=False)
            # sigma_y = tf.reduce_mean(tf.expand_dims(atts[5:10,:,:,:,i],4),axis=[0], keep_dims=False)
            # adain = (1+sigma_y)*x_+mu_y
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        # out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
        return adain_x
    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = tcl.conv2d(x, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=None,
                           # normalizer_fn=None)
                           normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
            # x = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # latent_in = tf.reshape(latent_in, (-1, self.y_num, input_shape[3]*self.f_num))
            latent_in = tf.reshape(latent_in, (-1, self.y_num, self.channel*self.f_num))
            # out = style_mod(x, latent_in, use_bias=True)
            style_xs = []
            for i in range(self.y_num):
                style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
                style_xs.append(style_x)
            style_xs = tf.concat(style_xs, axis=3)
            # out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, normalizer_fn=None)
            return style_xs
    def do_att(self, inputs, embs, is_training=True):#multi

        down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
                                 normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def do_mynorm_1(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)

        mu_y = atts[0]
        sigma_y = atts[1]
        # mu_y, sigma_y = tf.nn.moments(atts, axes=[0], keep_dims=True)

        adain = sigma_y * x_ + mu_y
        # adain = (1+sigma_y) * x_ + mu_y
        return adain
    def layer_adain_1(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = tcl.conv2d(x, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=None,
                           normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
                           # normalizer_fn=None)
            # x = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # latent_in = tf.reshape(latent_in, (-1, self.channel))
            out = style_mod(x, latent_in, use_bias=True)
            return out
    def do_att_1(self, inputs, embs_in, name=None, is_training=True):#only z, no label

        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
        embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
        embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            # att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
        return atts

    def __call__(self, inputs, embs_l, embs_l_s, embs_n, embs_n_s, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel), initializer=tf.initializers.zeros(), trainable=False)
            emb_n_avg = tf.get_variable('embs_n_avg', shape=(1, self.channel), initializer=tf.initializers.zeros(), trainable=False)

            with tf.variable_scope('zlAvg'):
                batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)
            with tf.variable_scope('znAvg'):
                batch_n_avg = tf.reduce_mean(embs_n, axis=0, keep_dims=True)
                update_op = tf.assign(emb_n_avg, lerp(batch_n_avg, emb_n_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_n = tf.identity(embs_n)
                embs_n = lerp(emb_n_avg, embs_n, self.trunc_psi)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de_r = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))
                sw_1 = tf.get_variable('l_multi_1_{}'.format(de.shape[1]), shape=(2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)

                #mynorm
                de = apply_noise(de_r, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                att = self.do_att_1(de, embs_n, name='att_1_{}'.format(de.shape[1]), is_training=is_training)  #id z
                de1 = self.do_mynorm_1(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                att = self.do_att(de, embs_l, is_training=is_training)  #condition
                de2 = self.do_mynorm(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de_my = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                #style
                de_s_1 = self.layer_adain(de_r, embs_l_s, training=is_training, name='style_l1_{}'.format(de.shape[1]))
                de_s_2 = self.layer_adain_1(de_r, embs_n_s, training=is_training, name='style_n1_{}'.format(de.shape[1]))
                de_concat = tf.concat((de_s_1, de_s_2), -1)
                de_style = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = de_my*sw_1[0] +de_style*sw_1[1]

                sw_2 = tf.get_variable('l_multi_2_{}'.format(de.shape[1]), shape=(2), initializer=tf.constant_initializer(value=2.0 / self.f_num), trainable=False)
                de_r = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                #mynorm
                de = apply_noise(de_r, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                att = self.do_att_1(de, embs_n, name='att_2_{}'.format(de.shape[1]), is_training=is_training)  #id z
                de1 = self.do_mynorm_1(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                de2 = self.do_mynorm(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de_my = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                #style
                de_s_1 = self.layer_adain(de_r, embs_l_s, training=is_training, name='style_l2_{}'.format(de.shape[1]))
                de_s_2 = self.layer_adain_1(de_r, embs_n_s, training=is_training, name='style_n2_{}'.format(de.shape[1]))
                de_concat = tf.concat((de_s_1, de_s_2), -1)
                de_style = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                de = de_my * sw_2[0] + de_style * sw_2[1]

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))


            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_id(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def do_mynorm(self, inputs, atts, name=None, is_training = True):
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)

        # sw = tf.get_variable('l_multi_'+name, shape=(self.f_num/2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)

        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            # mu_y = tf.reduce_mean(tf.expand_dims(atts[0:5,:,:,:,i],4), axis=0, keep_dims=False)
            # sigma_y = tf.reduce_mean(tf.expand_dims(atts[5:10,:,:,:,i],4),axis=[0], keep_dims=False)
            # mu_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
            # sigma_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
            # for j in range(self.f_num/2):
            #     mu_y += sw[j]*tf.expand_dims(atts[2*j,:,:,:,i],3)
            #     sigma_y += sw[j]*tf.expand_dims(atts[2*j+1,:,:,:,i],3)
            # adain = (1+sigma_y)*x_+mu_y
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        # out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
        #                          # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
        #                          normalizer_fn=None)
        # out = tf.reduce_mean(adains, axis=0)
        return adain_x

    def do_att(self, inputs, embs, is_training=True):#multi

        down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
                                 normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def do_mynorm_1(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)

        # sw = tf.get_variable('z_multi_'+name, shape=(self.f_num/2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)
        # mu_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
        # sigma_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
        # for j in range(self.f_num/2):
        #     mu_y += sw[j]*atts[2*j]
        #     sigma_y += sw[j]*atts[2*j+1]
        mu_y = atts[0]
        sigma_y = atts[1]
        # mu_y, sigma_y = tf.nn.moments(atts, axes=[0], keep_dims=True)

        adain = sigma_y * x_ + mu_y
        # adain = (1+sigma_y) * x_ + mu_y
        return adain

    def do_att_1(self, inputs, embs_in, name=None, is_training=True):#only z, no label

        input_shape = inputs.shape.as_list()
        # if input_shape[-1]!=32:
        # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
        #                          normalizer_fn=None)
        # else:
        #     down_inputs = inputs
        # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)

        embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
        embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)
        # embs = tf.reshape(embs_in, (-1, input_shape[3], self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            # att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
        return atts

    def __call__(self, inputs, embs_l, embs_n, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            # de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            # de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))


            # de = const
            # w = 4
            # with tf.name_scope('fc8') as scope:
            #     out = tcl.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            #
            # with tf.name_scope('fc6') as scope:
            #     out = tcl.fully_connected(out, w * w * int(self.get_nf(1)), activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            # de = tf.reshape(out, (-1, w, w, int(self.get_nf(1))))  # 4*4*512

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
                                      initializer=tf.initializers.zeros(), trainable=False)

            emb_n_avg = tf.get_variable('embs_n_avg', shape=(1, self.channel), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)
            with tf.variable_scope('znAvg'):
                batch_n_avg = tf.reduce_mean(embs_n, axis=0, keep_dims=True)
                update_op = tf.assign(emb_n_avg, lerp(batch_n_avg, emb_n_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_n = tf.identity(embs_n)
                embs_n = lerp(emb_n_avg, embs_n, self.trunc_psi)


            input_const = tf.get_variable("const", shape=[1, 16, 16, self.channel*4], initializer=tf.initializers.zeros())
            de = tf.tile(tf.cast(input_const, dtype=tf.float32), [shape[0], 1, 1, 1])

            de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
            # if i < pg-3:
            att = self.do_att_1(de, embs_n, name='att_1_{}'.format(de.shape[1]), is_training=is_training)  # id z
            # de1 = self.do_mynorm_1(de, att, is_training=is_training)
            de1 = self.do_mynorm_1(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
            # else:
            att = self.do_att(de, embs_l, is_training=is_training)  # condition z
            # de2 = self.do_mynorm(de, att, is_training=is_training)
            de2 = self.do_mynorm(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
            de_concat = tf.concat((de1, de2), -1)
            de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1,
                            activation_fn=tf.nn.leaky_relu, normalizer_fn=None)


            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                # de = lrelu(
                #     conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))
                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                # if i < pg-3:
                att = self.do_att_1(de, embs_n, name='att_1_{}'.format(de.shape[1]), is_training=is_training)  #id z
                # de1 = self.do_mynorm_1(de, att, is_training=is_training)
                de1 = self.do_mynorm_1(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                # else:
                att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                # de2 = self.do_mynorm(de, att, is_training=is_training)
                de2 = self.do_mynorm(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 normalizer_fn=None)
                # att = self.do_att(de, embs, name='att_1_{}'.format(de.shape[1]), is_training=is_training)
                # de = self.do_mynorm(de, att, is_training=is_training)

                # de = lrelu(
                #     conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))
                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                # if i < pg-3:
                att = self.do_att_1(de, embs_n, name='att_2_{}'.format(de.shape[1]), is_training=is_training)  #id z
                # de1 = self.do_mynorm_1(de, att, is_training=is_training)
                de1 = self.do_mynorm_1(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                # else:
                att = self.do_att(de, embs_l, is_training=is_training)  #condition z
                # de2 = self.do_mynorm(de, att, is_training=is_training)
                de2 = self.do_mynorm(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 normalizer_fn=None)
                # att = self.do_att(de, embs, name='att_2_{}'.format(de.shape[1]), is_training=is_training)
                # de = self.do_mynorm(de, att, is_training=is_training)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_id_new(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def do_mynorm(self, inputs, atts, name=None, is_training = True):
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs-mu_x)/(sigma_x+1e-6)

        # sw = tf.get_variable('l_multi_'+name, shape=(self.f_num/2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)

        adains = []
        for i in range(self.y_num):
            # att = tf.expand_dims(atts[:,:,:,i],3)
            # mu_y, sigma_y = tf.nn.moments(att, axes=[1,2], keep_dims=True)
            mu_y = tf.expand_dims(atts[0,:,:,:,i],3)
            sigma_y = tf.expand_dims(atts[1,:,:,:,i],3)
            # mu_y = tf.reduce_mean(tf.expand_dims(atts[0:5,:,:,:,i],4), axis=0, keep_dims=False)
            # sigma_y = tf.reduce_mean(tf.expand_dims(atts[5:10,:,:,:,i],4),axis=[0], keep_dims=False)
            # mu_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
            # sigma_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
            # for j in range(self.f_num/2):
            #     mu_y += sw[j]*tf.expand_dims(atts[2*j,:,:,:,i],3)
            #     sigma_y += sw[j]*tf.expand_dims(atts[2*j+1,:,:,:,i],3)
            # adain = (1+sigma_y)*x_+mu_y
            adain = sigma_y*x_+mu_y
            adains.append(adain)
        adain_x = tf.concat(adains, axis=3)
        # out = tcl.conv2d(adain_x, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
        #                          # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
        #                          normalizer_fn=None)
        # out = tf.reduce_mean(adains, axis=0)
        return adain_x

    def do_att(self, inputs, embs, is_training=True):#multi

        # down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
        #                          # normalizer_fn=tcl.batch_norm,  normalizer_params={'scale': True, 'is_training': is_training})
        #                          normalizer_fn=None)
        input_shape = inputs.shape.as_list()
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        att_w = []
        for i in range(self.f_num):
            input_r_ = tf.tile(tf.expand_dims(input_r, 0), (self.y_num,1,1,1))  # (40, N, size**2, c)
            emb = embs[i,:,:,:] #(f_num, N, 40, c)  # [N,40,c]
            emb = tf.transpose(emb, (1,0,2))  # [N,40,c]->[40,N,c]
            emb = tf.expand_dims(emb, 3)  #[40,N,c,1]
            att = tf.matmul(input_r_, emb)  # (40, N, size**2, 1)
            # att_weights = tf.nn.softmax(att, axis=0)  # (40, N, size**2, 1)
            # att = tf.nn.softmax(att, axis=3)  # (40, N, size**2, 1)
            att_w_soft = tf.reshape(tf.transpose(tf.squeeze(att), (1, 2, 0)), (-1, input_shape[1], input_shape[2], self.y_num))  # (N, size,size, 40)
            att_w.append(tf.expand_dims(att_w_soft,0))
        att_w = tf.concat(att_w, axis=0)   # (f_num, N, size,size, 40)
        # att_w = tf.reduce_mean(att_w, axis=0)  # (N, size,size, 40)
        return att_w

    def do_mynorm_1(self, inputs, atts, name=None, is_training=False):  # only z, no label
        input_shape = inputs.shape.as_list()
        # mu_x, sigma_x = tf.nn.moments(inputs, axes=[0,1,2], keep_dims=True)
        mu_x, sigma_x = tf.nn.moments(inputs, axes=[3], keep_dims=True)
        x_ = (inputs - mu_x) / (sigma_x + 1e-6)

        # sw = tf.get_variable('z_multi_'+name, shape=(self.f_num/2), initializer=tf.constant_initializer(value=2.0/self.f_num), trainable=False)
        # mu_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
        # sigma_y = tf.zeros((input_shape[0], input_shape[1], input_shape[2], 1))
        # for j in range(self.f_num/2):
        #     mu_y += sw[j]*atts[2*j]
        #     sigma_y += sw[j]*atts[2*j+1]
        mu_y = atts[0]
        sigma_y = atts[1]

        adain = sigma_y * x_ + mu_y
        # adain = (1+sigma_y) * x_ + mu_y
        return adain

    def do_att_1(self, inputs, embs, name=None, is_training=True):#only z, no label
        input_shape = inputs.shape.as_list()
        # down_inputs = tcl.conv2d(inputs, num_outputs=32, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
        input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        # embs = tf.reshape(embs_in, (-1, input_shape[3], self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            # att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)), axis=1)
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
        # atts = tf.concat(atts, axis=3)  # (N, size, size, self.f_num)
        return atts

    def __call__(self, inputs, embs_l, embs_n, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            # de = const
            # w = 4
            # with tf.name_scope('fc8') as scope:
            #     out = tcl.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            #
            # with tf.name_scope('fc6') as scope:
            #     out = tcl.fully_connected(out, w * w * int(self.get_nf(1)), activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            # de = tf.reshape(out, (-1, w, w, int(self.get_nf(1))))  # 4*4*512

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
                                      initializer=tf.initializers.zeros(), trainable=False)

            emb_n_avg = tf.get_variable('embs_n_avg', shape=(1, self.channel, self.f_num), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)
            with tf.variable_scope('znAvg'):
                batch_n_avg = tf.reduce_mean(embs_n, axis=0, keep_dims=True)
                update_op = tf.assign(emb_n_avg, lerp(batch_n_avg, emb_n_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_n = tf.identity(embs_n)
                embs_n = lerp(emb_n_avg, embs_n, self.trunc_psi)


            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                down_inputs = tcl.conv2d(de, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # input_r = tf.reshape(down_inputs, (-1, de.shape[1] * de.shape[2], self.channel))  # (N, size**2, c)

                att = self.do_att_1(down_inputs, embs_n, name='att_1_{}'.format(de.shape[1]), is_training=is_training)  #id z
                de1 = self.do_mynorm_1(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                att = self.do_att(down_inputs, embs_l, is_training=is_training)  #condition z
                de2 = self.do_mynorm(de, att, name='norm_1_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 normalizer_fn=None)

                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                down_inputs = tcl.conv2d(de, num_outputs=self.channel, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                # input_r = tf.reshape(down_inputs, (-1, de.shape[1] * de.shape[2], self.channel))  # (N, size**2, c)

                att = self.do_att_1(down_inputs, embs_n, name='att_2_{}'.format(de.shape[1]), is_training=is_training)  #id z
                de1 = self.do_mynorm_1(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                att = self.do_att(down_inputs, embs_l, is_training=is_training)  #condition z
                de2 = self.do_mynorm(de, att, name='norm_2_{}'.format(de.shape[1]), is_training=is_training)
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu,
                                 normalizer_fn=None)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_id_style(object):
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
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_bias=True)
            # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
            #                          normalizer_params={'scale': True, 'is_training': training})
            # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

            style_xs = []
            for i in range(self.y_num):
                style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
                style_xs.append(style_x)
            out = tf.concat(style_xs, axis=3)
            # out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1,
            #                  normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
            return out

    def layer_adain_1(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # style_xs = []
            # for i in range(self.y_num):
            #     style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
            #     style_xs.append(style_x)
            # style_xs = tf.concat(style_xs, axis=3)
            # out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1,
            #                  normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
            return out

    def __call__(self, inputs, embs_l, embs_n, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))


            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, self.y_num, self.channel),
                                      initializer=tf.initializers.zeros(), trainable=False)
            emb_n_avg = tf.get_variable('embs_n_avg', shape=(1, self.channel), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs_l = lerp(emb_l_avg, embs_l, self.trunc_psi)
            with tf.variable_scope('znAvg'):
                batch_n_avg = tf.reduce_mean(embs_n, axis=0, keep_dims=True)
                update_op = tf.assign(emb_n_avg, lerp(batch_n_avg, emb_n_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_n = tf.identity(embs_n)
                embs_n = lerp(emb_n_avg, embs_n, self.trunc_psi)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))

                de1 = self.layer_adain(de, embs_l, training=is_training, name='style_l_1_{}'.format(de.shape[1]))
                de2 = self.layer_adain_1(de, embs_n, training=is_training, name='style_n_1_{}'.format(de.shape[1]))
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                de1 = self.layer_adain(de, embs_l, training=is_training, name='style_l_2_{}'.format(de.shape[1]))
                de2 = self.layer_adain_1(de, embs_n, training=is_training, name='style_n_2_{}'.format(de.shape[1]))
                de_concat = tf.concat((de1, de2), -1)
                de = tcl.conv2d(de_concat, num_outputs=de.shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_style(object):
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
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, inputs, embs, pg=5, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            # inputs = tf.concat((inputs, labels), 1)
            # de = tf.reshape(Pixl_Norm(inputs), [input_shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale,
            #             gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [input_shape[0], 4, 4, int(self.get_nf(1))])
            # de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))

            # de = const
            w = 7
            with tf.name_scope('fc8') as scope:
                out = tcl.fully_connected(inputs, 1024, activation_fn=tf.nn.leaky_relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})

            with tf.name_scope('fc6') as scope:
                out = tcl.fully_connected(out, w * w * int(self.get_nf(1)), activation_fn=tf.nn.leaky_relu, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
            de = tf.reshape(out, (-1, w, w, int(self.get_nf(1))))  # 4*4*512

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)

                de = lrelu(conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))
                de = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)

                de = lrelu(conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))
                de = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class stylegangenerate_style_res(object):
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

    def __call__(self, inputs, embs, pg=5, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            # inputs = tf.concat((inputs, labels), 1)
            # de = tf.reshape(Pixl_Norm(inputs), [input_shape[0], 1, 1, -1])
            # de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            # de = Pixl_Norm(lrelu(de))
            # de = tf.reshape(de, [input_shape[0], 4, 4, int(self.get_nf(1))])
            # de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            # de = Pixl_Norm(lrelu(de))

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)
            # emb_avg = tf.get_variable('latent_avg', shape=(1, self.y_num, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)

            # w = 4
            # with tf.name_scope('fc8') as scope:
            #     out = tcl.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            #
            # with tf.name_scope('fc6') as scope:
            #     out = tcl.fully_connected(out, w * w * int(self.get_nf(1)), activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #                               normalizer_params={'scale': True, 'is_training': is_training})
            # de = tf.reshape(out, (-1, w, w, int(self.get_nf(1))))  # 4*4*512

            # w = 7
            # input_const = tf.get_variable("const", shape=[1, 16, 16, self.channel*4], initializer=tf.initializers.zeros())
            # de = tf.tile(tf.cast(input_const, dtype=tf.float32), [input_shape[0], 1, 1, 1])

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                res = de
                res = conv2d(res, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='res_{}'.format(res.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='res%d'%i)
                res = lrelu(res)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))
                if is_mlp:
                    de = self.layer_adain(de, embs, training=is_training, name='style1%d'%i)
                de = lrelu(de)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))
                if is_mlp:
                    de = self.layer_adain(de, embs, training=is_training, name='style2%d'%i)
                de = lrelu(de)

                de = res + de

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class stylegangenerate_style_res1(object):
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

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_bias=True)
            # down_inputs = tcl.conv2d(x, num_outputs=16, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
            #                          normalizer_params={'scale': True, 'is_training': training})
            # input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], 16))  # (N, size**2, c)

            style_xs = []
            for i in range(self.y_num):
                style_x = style_mod(x, latent_in[:, i, :], use_bias=True, name='Style%d' % i)  # latent_in(N, 40, c)
                style_xs.append(style_x)
            style_xs = tf.concat(style_xs, axis=3)
            out = tcl.conv2d(style_xs, num_outputs=input_shape[-1], kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=None)
                             # normalizer_fn=tcl.batch_norm, normalizer_params={'scale': True, 'is_training': training})
            return out
    # def layer_adain(self, x, latent_in, training=False, name=None):
    #     with tf.variable_scope(name or 'adain') as scope:
    #         input_shape = x.shape.as_list()
    #         # x = apply_noise(x, noise_var=None, random_noise=True)
    #         x = lrelu(apply_bias(x))
    #         x = myinstance_norm(x)
    #
    #         out = style_mod(x, latent_in, use_bias=True)
    #         return out

    def __call__(self, inputs, en_embs, embs, pg=5, t=False, alpha_trans=0.0, labels=None, is_mlp=False, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            # inputs = tf.concat((inputs, labels), 1)
            de = tf.reshape(Pixl_Norm(inputs), [input_shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2) / 4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [input_shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            emb_avg = tf.get_variable('latent_avg', shape=(1, embs.shape[1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('DlatentAvg'):
                batch_avg = tf.reduce_mean(embs, axis=0, keep_dims=True)
                update_op = tf.assign(emb_avg, lerp(batch_avg, emb_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs = tf.identity(embs)
                embs = lerp(emb_avg, embs, self.trunc_psi)
            # a=embs[:, tf.newaxis, :]
            # en_embs = Pixl_Norm(en_embs)
            embs_in = tf.concat((embs[:,tf.newaxis,:], en_embs[:,tf.newaxis,:]), 1)

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                res = de
                res = conv2d(res, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='res_{}'.format(res.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs_in, training=is_training, name='res%d'%i)
                # res = lrelu(res)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))
                if is_mlp:
                    de = self.layer_adain(de, embs_in, training=is_training, name='style1%d'%i)
                # de = lrelu(de)

                de = conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))
                if is_mlp:
                    de = self.layer_adain(de, embs_in, training=is_training, name='style2%d'%i)
                # de = lrelu(de)

                de = res + de

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            # if pg == 1: return de
            # if t: de = (1 - alpha_trans) * de_iden + alpha_trans*de
            # else: de = de

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegangenerate_doatt(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, channel=32, y_num=40, y_dim=2, f_num=2):
        self.name = 'stylegangenerate'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.f_num = f_num
        self.channel = channel

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def do_att(self, inputs, embs_in, name=None, is_training=True):#only z, no label
        down_inputs = tcl.conv2d(inputs, num_outputs=self.channel, kernel_size=1, stride=1, normalizer_fn=tcl.batch_norm,
                                          normalizer_params={'scale': True, 'is_training': is_training})
        input_shape = inputs.shape.as_list()
        # input_r = tf.reshape(inputs, (-1, input_shape[1] * input_shape[2], input_shape[3]))  # (N, size**2, c)
        input_r = tf.reshape(down_inputs, (-1, input_shape[1] * input_shape[2], self.channel))  # (N, size**2, c)

        # embs = lrelu(fully_connected(embs_in, input_shape[3]*self.f_num, use_bias=True, scope=name))
        # embs = tf.reshape(embs, (-1, input_shape[3], self.f_num))  # (N, c, f_num)
        embs = tf.reshape(embs_in, (-1, self.channel, self.f_num))  # (N, c, f_num)

        atts = []
        for i in range(self.f_num):
            emb = tf.expand_dims(embs[:, :, i], 2)  # (N, c, 1)
            att = tf.matmul(input_r, emb)  # (N, size**2, 1)
            # att = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))
            att = tf.nn.softmax(tf.reshape(att, (-1, input_shape[1] ** 2)))
            att_weight = tf.reshape(att, (-1, input_shape[1], input_shape[2], 1))  # (N, size, size, 1)
            atts.append(att_weight)
            out = tf.multiply(inputs, att_weight)
            out = out + inputs
        return out

    def __call__(self, inputs, embs, pg=5, t=False, alpha_trans=0.0, is_training = False, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse == True:
                scope.reuse_variables()

            input_shape = inputs.shape.as_list()
            de = tf.reshape(Pixl_Norm(inputs), [input_shape[0], 1, 1, -1])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [input_shape[0], 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            # de = tf.get_variable('const', shape=[input_shape[0], 4, 4, int(self.get_nf(1))], initializer=tf.initializers.ones())

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1])))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_1_{}'.format(de.shape[1]))
                de = self.do_att(de, embs, name='att_1_{}'.format(de.shape[1]), is_training=is_training)
                # de = self.do_mynorm(de, att, is_training=is_training)

                de = lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1])))

                de = apply_noise(de, noise_var=None, random_noise=True, name='noise_2_{}'.format(de.shape[1]))
                de = self.do_att(de, embs, name='att_2_{}'.format(de.shape[1]), is_training=is_training)
                # de = self.do_mynorm(de, att, is_training=is_training)

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            return de
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class styleganencoder_concat(object):
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

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def __call__(self, conv, batch_size, reuse=False, pg=5, t=False, alpha_trans=0.01, labels=None, is_training = False):
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

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_res_{}'.format(res.shape[1]))
                res = batch_norm(res, scope='batch_norm_res_{}'.format(res.shape[1]))
                res = lrelu(res)

                conv = conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1]))
                conv = batch_norm(conv, scope='batch_norm_1_{}'.format(conv.shape[1]))
                conv = lrelu(conv)

                conv = conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_2_{}'.format(conv.shape[1]))
                conv = batch_norm(conv, scope='batch_norm_2_{}'.format(conv.shape[1]))
                conv = lrelu(conv)

                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            # output = fully_connected(conv, 1, use_bias=False, sn=False, scope='o_1')
            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class styleganencoder_style(object):
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

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            # x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, is_mlp=False, reuse=False, pg=5, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()
            # for i in range(self.y_num):
            #     label = labels[:, i, :]  # [N, 2]
            #     y = tf.reshape(label, (-1, 1, 1, 2))
            #     y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            #     conv = tf.concat([conv, y], 3)

            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
            #                             initializer=tf.initializers.zeros(), trainable=False)

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_res_{}'.format(conv.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                     name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                                       name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d' % i)
                conv = lrelu(conv)

                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden


            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            # conv = tf.reshape(conv, [batch_size, -1])
            out = tf.reduce_mean(conv, axis=(1, 2))


            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None, biases_initializer=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None, biases_initializer=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class styleganencoder_style_spatial(object):
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

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            out = style_mod(x, latent_in, use_bias=True)
            # out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, is_mlp=False, reuse=False, pg=5, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
            #                             initializer=tf.initializers.zeros(), trainable=False)

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_res_{}'.format(conv.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                     name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                                       name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d' % i)
                conv = lrelu(conv)

                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden


            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            # conv = lrelu(
            #     conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            # conv = tf.reshape(conv, [batch_size, -1])
            # out = tf.reduce_mean(conv, axis=(1, 2))


            with tf.name_scope('fc8') as scope:
                # z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None, biases_initializer=None)
                # z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None, biases_initializer=None)
                z_mu = conv_2d(conv, channels=self.z_dim, kernel=3, stride=1, pad=1, sn=False, name='enc_mu')
                z_mu = tf.reduce_mean(z_mu, axis=3, keep_dims=True)
                z_logvar = conv_2d(conv, channels=self.z_dim, kernel=3, stride=1, pad=1, sn=False, name='enc_sigma')
                z_logvar = tf.reduce_mean(z_logvar, axis=3, keep_dims=True)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class styleganencoder_style_ogan(object):
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

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            # x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_bias=True)
            out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, is_mlp=False, reuse=False, pg=5, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()

            y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            conv = tf.concat([conv, y], 3)

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_res_{}'.format(conv.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                     name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                                       name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d' % i)
                conv = lrelu(conv)

                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            #for D
            # output = tcl.fully_connected(conv, 1,  activation_fn=None,  biases_initializer=None, scope='o_1')
            # output = fully_connected(conv, 1, use_bias=False, sn=False, scope='o_1')
            out = lrelu(fully_connected(conv, 1024, use_bias=True, sn=False, scope='o_1'))
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            out = lrelu(fully_connected(out, self.z_dim, use_bias=False, sn=False, scope='code'))
            y_cls = lrelu(fully_connected(out, self.y_dim, use_bias=False, sn=False, scope='cls'))
            return out, y_cls

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class stylegan_style_block(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2, latent_beta=0.995, trunc_psi=0.7):
        self.name = 'stylegan_style'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num
        self.latent_beta = latent_beta
        self.trunc_psi = trunc_psi

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def layer_adain(self, x, latent_in, training=False, name=None):
        with tf.variable_scope(name or 'adain') as scope:
            input_shape = x.shape.as_list()
            x = apply_noise(x, noise_var=None, random_noise=True)
            x = lrelu(apply_bias(x))
            x = myinstance_norm(x)

            # out = style_mod(x, latent_in, use_bias=True)
            out = style_mod_rev(x, latent_in, use_bias=True)
            return out

    def __call__(self, conv, embs_l, batch_size, is_mlp=False, reuse=False, pg=5, t=False, alpha_trans=0.01, labels=None, is_training = False):
        with tf.variable_scope(self.name) as scope:

            # fromRGB
            input_shape = conv.shape.as_list()
            # y = tf.reshape(labels, (-1, 1, 1, self.y_dim))
            # y = tf.tile(y, (1, input_shape[1], input_shape[1], 1))
            # conv = tf.concat([conv, y], 3)

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,sn=True,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))

            # emb_l_avg = tf.get_variable('embs_l_avg', shape=(self.f_num, 1, self.y_num, self.channel),
            #                             initializer=tf.initializers.zeros(), trainable=False)

            emb_l_avg = tf.get_variable('embs_l_avg', shape=(1, embs_l.shape[-1]), initializer=tf.initializers.zeros(), trainable=False)

            # Update moving average of W.
            with tf.variable_scope('zlAvg'):
                # batch_l_avg = tf.reduce_mean(embs_l, axis=1, keep_dims=True)
                batch_l_avg = tf.reduce_mean(embs_l, axis=0, keep_dims=True)
                update_op = tf.assign(emb_l_avg, lerp(batch_l_avg, emb_l_avg, self.latent_beta))
                with tf.control_dependencies([update_op]):
                    embs_l = tf.identity(embs_l)
                embs = lerp(emb_l_avg, embs_l, self.trunc_psi)

            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, sn=False,name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                res = conv
                res = conv2d(res, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_res_{}'.format(conv.shape[1]))
                if is_mlp:
                    res = self.layer_adain(res, embs, training=is_training, name='style_res%d'%i)
                res = lrelu(res)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                     name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style1%d'%i)
                conv = lrelu(conv)

                # conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                #                                       name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,sn=False,
                                    name='dis_n_conv_2_{}'.format(conv.shape[1]))
                if is_mlp:
                    conv = self.layer_adain(conv, embs, training=is_training, name='style2%d' % i)
                conv = lrelu(conv)

                conv = res + conv

                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden


            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale,
                       sn=True, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale,
                       sn=True, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [batch_size, -1])

            return conv

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class styleganencoder_style_block(object):
    #dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2):
        self.name = 'styleganencoder'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num

    def get_nf(self, stage):
        return min(512 / (2 **(stage * 1)), 256)

    def __call__(self, conv, out_share, batch_size, reuse=False, labels=None, is_mlp=False, is_training = False):
        with tf.variable_scope(self.name) as scope:

            if reuse == True:
                scope.reuse_variables()
            out = out_share

            # conv = lrelu(
            #     conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,name='dis_n_conv_1_{}'.format(conv.shape[1])))
            # conv = lrelu(
            #     conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, sn=False,padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            # conv = tf.reshape(conv, [batch_size, -1])

            # out = lrelu(fully_connected(out, 1024, use_bias=True, sn=False, scope='o_1'))

            with tf.name_scope('fc8') as scope:
                z_mu = tcl.fully_connected(out, self.z_dim, activation_fn=None)
                z_logvar = tcl.fully_connected(out, self.z_dim, activation_fn=None)
            return z_mu, z_logvar

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class stylegandiscriminate_style_block(object):
    # dis_as_v = []
    def __init__(self, z_dim=256, y_num=40, y_dim=2, channel=32, f_num=2):
        self.name = 'stylegandiscriminator'
        self.z_dim = z_dim
        self.use_wscale = False
        self.y_num = y_num
        self.y_dim = y_dim
        self.channel = channel
        self.f_num = f_num

    def get_nf(self, stage):
        return min(512 / (2 ** (stage * 1)), 256)

    def __call__(self, out_share, reuse=False, labels=None, is_training=False):
        with tf.variable_scope(self.name) as scope:

            if reuse == True:
                scope.reuse_variables()
            out = out_share

            # conv = MinibatchstateConcat(conv)
            # conv = lrelu(
            #     conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale,
            #            sn=True, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            # conv = lrelu(
            #     conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale,
            #            sn=True, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            # out = tf.reshape(conv, [batch_size, -1])

            output = fully_connected(out, 1, use_bias=False, sn=True, scope='o_1')
            # out = lrelu(fully_connected(out, 1024, sn=True, scope='o_0'))

            # y_cls = fully_connected(out, self.y_dim, use_bias=False, sn=True, scope='cls')

            # return output, y_cls
            return output

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)