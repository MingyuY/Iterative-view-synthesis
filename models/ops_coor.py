import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.image import dense_image_warp
from skimage.measure import compare_ssim
import cv2
import matplotlib.pyplot as plt
import rnn_cell
from tensorflow.python.ops import math_ops
#weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_init = xavier_initializer()
# weight_init = variance_scaling_initializer()

import math
# weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
weight_regularizer = None

# pad = (k-1) // 2 = SAME !
# output = ( input - k + 1 + 2p ) // s 
#def gaussian(x, mu, log_var):
#    import math
##    gaussian = tf.exp(-(x - mu) ** 2 / (2 * tf.exp(log_var)+ 1e-6)) / (tf.sqrt(2 * math.pi * tf.exp(log_var))+ 1e-6)
#    gaussian = tf.exp(-(x - mu) ** 2 / (2 * tf.exp(log_var)+ 1e-6)) / (tf.sqrt(2 * math.pi * tf.exp(log_var))+ 0)
#    return gaussian
def gaussian(x, mu, log_var):
    import math
    gaussian = tf.exp(-(x - mu) ** 2 / (2 * tf.exp(log_var))) / (tf.sqrt(2 * math.pi * tf.exp(log_var)))
    return gaussian

def log_density(z, y, mu, log_var):
    z_shape = z.shape.as_list()
    # density = norm.pdf(code, loc=mu, scale=sigma)
    # density_map = tf.exp(-(x - mu) ** 2 / (2 * tf.exp(log_var))) / (tf.sqrt(2 * math.pi * log_var))
    factor = 20
    density_map = gaussian(z, mu, log_var) * factor
    density = tf.reduce_sum(tf.reduce_prod(density_map, axis=[1 ]))
#    density = tf.reduce_sum(tf.reduce_prod(density_map, axis=[1, 2, 3]))
    # log_density = tf.reduce_sum(density_map, axis=[1, 2, 3])
    # density = tf.exp(log_density)

    c_sets = []
    probs = []
    num = []
    d1 = []
    d2 = []
    # i=0
    for i in range(z_shape[0]):
        c = y[i]
        c_set = tf.to_float(tf.equal(y, c))
        z_batch = tf.multiply(tf.ones_like(z, dtype=tf.float32), z[i])
        c_sets.append(c_set)

        count_c_num = tf.reduce_sum(c_set)
        density_z_all_ = gaussian(z_batch, mu, log_var) * factor
#        density_z_all = tf.reduce_prod(density_z_all, axis=[1,2,3 ])
        density_z_all = tf.reduce_prod(density_z_all_, axis=[1 ])
        density_z_under_c = tf.reduce_sum(tf.multiply(density_z_all, c_set)) / count_c_num  # q(z|c)
        # d1.append(density_z_all)
        # d2.append(density_z_under_c)
        # num.append(count_c_num)
        density_z = tf.reduce_mean(density_z_all)  # q(z)
        prob = tf.log(density_z_under_c / density_z)
        probs.append(prob)
    probs = tf.convert_to_tensor(probs)
    _, idx = tf.unique(y)  # (y, idx)
    probs_c = []
    for k in range(z_shape[0]):
        position = tf.to_float(tf.equal(idx, k))  #  
        c_num = tf.reduce_sum(position)
        prob_c = tf.cond(tf.greater(c_num, 0.0), lambda: tf.reduce_sum(tf.multiply(probs, position)) / c_num,
                         lambda: 0.0)
        # prob_c = tf.reduce_sum(tf.multiply(probs, position))/c_num
        probs_c.append(prob_c)
    probs_c = tf.convert_to_tensor(probs_c)
    category = tf.count_nonzero(probs_c, dtype=tf.float32)  # 
    out = tf.reduce_sum(probs_c) / category

    return out, density_z_under_c, density_z, density_z_all_

#def log_density(z, y, mu, log_var):
#    z_shape = z.shape.as_list()
#    # density = norm.pdf(code, loc=mu, scale=sigma)
#    # density_map = tf.exp(-(x - mu) ** 2 / (2 * tf.exp(log_var))) / (tf.sqrt(2 * math.pi * log_var))
#    factor = 4
#    density_map = gaussian(z, mu, log_var) * factor
#    density = tf.reduce_sum(tf.reduce_prod(density_map, axis=[1 ]))
##    density = tf.reduce_sum(tf.reduce_prod(density_map, axis=[1, 2, 3]))
#    # log_density = tf.reduce_sum(density_map, axis=[1, 2, 3])
#    # density = tf.exp(log_density)
#
#    c_sets = []
#    probs = []
#    num = []
#    d1 = []
#    d2 = []
#    # i=0
#    for i in range(z_shape[0]):
#        c = y[i]
#        c_set = tf.to_float(tf.equal(y, c))
#        z_batch = tf.multiply(tf.ones_like(z, dtype=tf.float32), z[i])
#        c_sets.append(c_set)
#
#        count_c_num = tf.reduce_sum(c_set)
#        density_z_all = gaussian(z_batch, mu, log_var) * factor
##        density_z_all = tf.reduce_prod(density_z_all, axis=[1,2,3 ])
#        density_z_all = tf.reduce_prod(density_z_all, axis=[1 ])
##        density_z_under_c = tf.reduce_sum(tf.multiply(density_z_all, c_set)) / (count_c_num+ 1e-6)  # q(z|c)
#        density_z_under_c = tf.reduce_sum(tf.multiply(density_z_all, c_set)) / (count_c_num+ 0)  # q(z|c)
#        # d1.append(density_z_all)
#        # d2.append(density_z_under_c)
#        # num.append(count_c_num)
#        density_z = tf.reduce_mean(density_z_all)  # q(z)
#        prob = tf.log(density_z_under_c / (density_z + 0) + 0)
##        prob = tf.log(density_z_under_c / (density_z + 1e-6) + 1e-6)
#        probs.append(prob)
#    probs = tf.convert_to_tensor(probs)
#    _, idx = tf.unique(y)  # (y, idx)
#    probs_c = []
#    for k in range(z_shape[0]):
#        position = tf.to_float(tf.equal(idx, k))  #  
#        c_num = tf.reduce_sum(position)
#        prob_c = tf.cond(tf.greater(c_num, 0.0), lambda: tf.reduce_sum(tf.multiply(probs, position)) / (c_num + 0),
##        prob_c = tf.cond(tf.greater(c_num, 0.0), lambda: tf.reduce_sum(tf.multiply(probs, position)) / (c_num + 1e-6),
#                         lambda: 0.0)
#        # prob_c = tf.reduce_sum(tf.multiply(probs, position))/c_num
#        probs_c.append(prob_c)
#    probs_c = tf.convert_to_tensor(probs_c)
#    category = tf.count_nonzero(probs_c, dtype=tf.float32)  # 
#    out = tf.reduce_sum(probs_c) / (category + 0)
##    out = tf.reduce_sum(probs_c) / (category + 1e-6)
#
#    return out, probs

def show_1flow_arrow(bgimg, array_YX, save_path, color = (0,0,255), inter=16):
    bgimg=(((bgimg+1)/2)*255.0).astype(np.uint8)
    bgimg_write=(((np.ones_like(bgimg)+1)/2)*255.0).astype(np.uint8)
    b, h, w, c = array_YX.shape
    b, h_img, w_img, _ = bgimg.shape
    imgs = []
    imgs_b = []
    imgs_bg = []
    flow_x_imgs = []
    flow_y_imgs = []
    flow_diff_imgs = []
    image_out = np.zeros_like(bgimg)
    out_boundbox = []
    if b>3:
        b=4
    for k in  range(b):
        flow = array_YX[k]*h_img/h
        flow = np.reshape(flow, [h, w, 2])
        flow_y_ = cv2.resize(np.repeat(np.expand_dims(flow[:,:,0], -1),3,-1), (h_img, w_img))
        flow_x_ = cv2.resize(np.repeat(np.expand_dims(flow[:,:,1], -1),3,-1), (h_img, w_img))
        flow_diff = flow_x_ - flow_y_
#        flow_diff_ = (((flow_diff - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_diff_imgs.append(flow_diff)
#        flow_y_1 = (((flow_y_ - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_y_imgs.append(flow_y_)
#        flow_x_1 = (((flow_x_ - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_x_imgs.append(flow_x_)
        image= bgimg[k].copy()
        image_write = bgimg_write[k].copy()
        s = 0
        for i in range(0, h_img, inter):
            for j in range(0, w_img, inter):#inter
                if not (int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])>(w_img-1) or \
                        int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])<0 or \
                        int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])>(h_img-1) or \
                        int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])<0):
                    img = cv2.arrowedLine(image_write, (j,i), (int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1]),\
                                                               int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])), color, 1, 8,0,0.2)
                    s = 1
 
        for i in range(0, h_img, 1):
            for j in range(0, w_img, 1):#inter
                if not (int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])>(w_img-1) or \
                        int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])<0 or \
                        int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])>(h_img-1) or \
                        int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])<0):
#                    img = cv2.arrowedLine(image_write, (j,i), (int(j-flow[j/h_img*h, i/w_img*w, 1]), int(i-flow[j/h_img*h, i/w_img*w, 0])), color, 1, 8,0,0.2)
                    s = 1
                    image_out[k][i, j] = np.ones_like(image_out[k][i, j])*255
#                    image_out[k][i, j] = image[int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0]), \
#                                               int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])]
 
                    
        if s==0:
            imgs.append(image)
            imgs_b.append(np.expand_dims(image, 0))
            imgs_bg.append(bgimg[k])
        else:
            imgs.append(img)
            imgs_b.append(np.expand_dims(img, 0))
            imgs_bg.append(bgimg[k])
        img = np.concatenate(imgs, 0)
        img_b = np.concatenate(imgs_b, 0)
        img_bg = np.concatenate(imgs_bg, 0)
        img_flowy = np.concatenate(flow_y_imgs, 0)
        img_flowx = np.concatenate(flow_x_imgs, 0)
        img_flowdiff = np.concatenate(flow_diff_imgs, 0)
    return img, img_b,img_bg, img_flowy, img_flowx, img_flowdiff, image_out
    
def show_flow_arrow(bgimg, array_YX, save_path, color = (0,0,255), inter=16):
    bgimg=(((bgimg+1)/2)*255.0).astype(np.uint8)
    bgimg_write=(((np.ones_like(bgimg)+1)/2)*255.0).astype(np.uint8)
    b, h, w, c = array_YX.shape
    b, h_img, w_img, _ = bgimg.shape
    imgs = []
    imgs_b = []
    imgs_bg = []
    flow_x_imgs = []
    flow_y_imgs = []
    flow_diff_imgs = []
    if b>3:
        b=4
    for k in  range(b):
        flow = array_YX[k]*h_img/h
        flow = np.reshape(flow, [h, w, 3, 3, 2])
        flow =  np.mean(flow, axis=-2)
        flow =  np.mean(flow, axis=-2)
#        flow_y = np.expand_dims( flow[:,:,8] , -1)
#        flow_x = np.expand_dims( flow[:,:,9] , -1)
#        flow_y = np.expand_dims(np.mean(flow[:,:,[2,6,8,10,14]], -1), -1)
#        flow_x = np.expand_dims(np.mean(flow[:,:,[3,7,9,11,15]], -1), -1)
#        flow_y = np.expand_dims(np.mean(flow[:,:,:9], -1), -1)
#        flow_x = np.expand_dims(np.mean(flow[:,:,9:], -1), -1)
#        flow = np.concatenate([flow_y, flow_x], -1)
        flow_y_ = cv2.resize(np.repeat(np.expand_dims(flow[:,:,0], -1),3,-1), (h_img, w_img))
        flow_x_ = cv2.resize(np.repeat(np.expand_dims(flow[:,:,1], -1),3,-1), (h_img, w_img))
        flow_diff = flow_x_ - flow_y_
#        flow_diff_ = (((flow_diff - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_diff_imgs.append(flow_diff)
#        flow_y_1 = (((flow_y_ - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_y_imgs.append(flow_y_)
#        flow_x_1 = (((flow_x_ - np.min([flow_y_, flow_x_]))/((np.max([flow_y_, flow_x_]) - np.min([flow_y_, flow_x_]))))*255.0).astype('uint8')
        flow_x_imgs.append(flow_x_)
        image= bgimg[k].copy()
        image_write = bgimg_write[k].copy()
        s = 0
        for i in range(0, h_img, inter):
            for j in range(0, w_img, inter):
                
                if not (int(j+flow[j/h_img*h, i/w_img*w, 1])>w_img or int(j+flow[j/h_img*h, i/w_img*w, 1])<0 or \
                   int(i+flow[j/h_img*h, i/w_img*w, 0])>h_img or int(i+flow[j/h_img*h, i/w_img*w, 0])<0):
                    img = cv2.arrowedLine(image_write, (j,i), (int(j+flow[j/h_img*h, i/w_img*w, 1]), int(i+flow[j/h_img*h, i/w_img*w, 0])), color, 1, 8,0,0.2)
                    s = 1
#                len = np.sqrt(np.square(flow[i, j, 0]) + np.square(flow[i, j, 1]))
#                plt.quiver(j, i, flow[i, j, 1], flow[i, j, 0], color='g', width=0.005)
        if s==0:
            imgs.append(image)
            imgs_b.append(np.expand_dims(image, 0))
            imgs_bg.append(bgimg[k])
#            img = np.concatenate(imgs, 0)
#            img_b = np.concatenate(imgs_b, 0)
#            img_bg = np.concatenate(imgs_bg)
        else:
            imgs.append(img)
            imgs_b.append(np.expand_dims(img, 0))
            imgs_bg.append(bgimg[k])
        img = np.concatenate(imgs, 0)
        img_b = np.concatenate(imgs_b, 0)
        img_bg = np.concatenate(imgs_bg, 0)
        img_flowy = np.concatenate(flow_y_imgs, 0)
        img_flowx = np.concatenate(flow_x_imgs, 0)
        img_flowdiff = np.concatenate(flow_diff_imgs, 0)
    return img, img_b,img_bg, img_flowy, img_flowx, img_flowdiff

def show_1paired_flow_arrow(bgimg, array_YX, Nor, mul=1, save_path='', color = (255,189,0), inter=4):
    bgimg0 = bgimg[0]
    bgimg1 = bgimg[1]
    bgimg0=(((bgimg0+1)/2)*255.0).astype(np.uint8)
    bgimg1=(((bgimg1+1)/2)*255.0).astype(np.uint8)
    bgimg=cv2.addWeighted(bgimg0, 0.0, bgimg1, 0.8, 0)

    bgimg_write=(((np.ones_like(bgimg)+1)/2)*255.0).astype(np.uint8)
    h, w, c = array_YX.shape
    h_img, w_img, _ = bgimg.shape
    imgs = []
    imgs_b = []
    imgs_bg = []
    flow_x_imgs = []
    flow_y_imgs = []
    flow_diff_imgs = []
    flow = array_YX*h_img/h
    flow = np.reshape(flow, [h, w, 2]) 
    
    image_write= bgimg.copy()
#    image_write = bgimg_write.copy()
    s = 0
    for i in range(16, 96, inter):
        for j in range(10, 120, inter):#inter
            if not (int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])>(w_img-1) or \
                    int(j-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])<0 or \
                    int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])>(h_img-1) or \
                    int(i-flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0])<0):
                x = flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1]
                y = flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0]
                lena = math.sqrt((x**2)+(y**2)) + 1e-10
                if lena<10:
                    img = cv2.arrowedLine(image_write, (j,i), (int(j-(flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1]/lena)*mul),\
                                                  int(i-(flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0]/lena)*mul)), color, 1 , 8, 0, 0.2)
                else:
                    img = cv2.arrowedLine(image_write, (j,i), (int(j-(flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 1])),\
                                                  int(i-(flow[int(j/float(h_img)*h), int(i/float(w_img)*w), 0]))), color, 1 , 8, 0, 0.2)
                
                s = 1 
                
#    for i in range(0, h_img, inter):
#        for j in range(0, w_img, inter):
#            
#            if not (int(j+flow[j/h_img*h, i/w_img*w, 1])>w_img or int(j+flow[j/h_img*h, i/w_img*w, 1])<0 or \
#               int(i+flow[j/h_img*h, i/w_img*w, 0])>h_img or int(i+flow[j/h_img*h, i/w_img*w, 0])<0):
#                img = cv2.arrowedLine(image_write, (j,i), (int(j+flow[j/h_img*h, i/w_img*w, 1]), int(i+flow[j/h_img*h, i/w_img*w, 0])), color, 1, 8,0,0.2)
#                s = 1
    if s==0:
        img=image
    return img

def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))
    
def viz_flow(flow): 
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr 

def conv2d_d(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)


def deform_loaclatt_conv2d(x, offset, filter_shape, activation = None, scope=None, alpha=1.0):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    offset_shape = offset.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    _, o_h, o_w, o_oc = offset_shape
    o_ic = i_c
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

#    with tf.variable_scope(scope or "deform_conv"):
#        offset_map = conv2d_d(x, offset_shape, padding=True, scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = offset
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
#    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, 2, f_h, f_w])
#    offset_map = tf.transpose(offset_map, [0,1,2, 4, 5, 3])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range( -(f_w/2), (f_w/2)+1, 1 , dtype=tf.float32), tf.range( -(f_h/2), (f_h/2)+1, 1 , dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip =  tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1, 2, 3, 0, 4, 5]) # [batch, i_h, f_h, i_w, f_w, i_c]
    
    x_ip_reshape = tf.reshape(x_ip, [batch*i_h*i_w, i_c, f_h*f_w])
    x_ip_transpose = tf.transpose(x_ip_reshape, [0,2,1])
    cov = tf.nn.softmax(alpha*tf.matmul(x_ip_transpose, x_ip_reshape))
    x_ip_new = tf.matmul(x_ip_reshape, cov)
    x_ip_new = tf.reshape(x_ip_new, [batch, i_h, i_w, i_c, f_h, f_w])
    x_ip_new = tf.transpose(x_ip_new, [0, 1, 4, 2, 5, 3])
#    x_ip_new = tf.reduce_mean(x_ip_new, -1)
    x_ip_new = tf.reshape(x_ip_new, [batch, i_h*f_h, i_w*f_w, i_c]) 
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d_d(x_ip_new, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv, offset_map

   
def our_deform_conv2d(x, offset, filter_shape, activation = None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    offset_shape = offset.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    _, o_h, o_w, o_oc = offset_shape
    o_ic = i_c
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

#    with tf.variable_scope(scope or "deform_conv"):
#        offset_map = conv2d_d(x, offset_shape, padding=True, scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = offset
#    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, 2, f_h, f_w])
    offset_map = tf.transpose(offset_map, [0,1,2, 4, 5, 3])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range( -(f_w/2), (f_w/2)+1, 1 , dtype=tf.float32), tf.range( -(f_h/2), (f_h/2)+1, 1 , dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d_d(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv
   
def deform_conv2d(x, offset, filter_shape, activation = None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    offset_shape = offset.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    _, o_h, o_w, o_oc = offset_shape
    o_ic = i_c
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    assert o_oc==2*f_h*f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

#    with tf.variable_scope(scope or "deform_conv"):
#        offset_map = conv2d_d(x, offset_shape, padding=True, scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = offset
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
#    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, 2, f_h, f_w])
#    offset_map = tf.transpose(offset_map, [0,1,2, 4, 5, 3])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range( -(f_w/2), (f_w/2)+1, 1 , dtype=tf.float32), tf.range( -(f_h/2), (f_h/2)+1, 1 , dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d_d(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv

def instance_norm(input, is_training):
    """ instance normalization """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def bn_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                # batch norm
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True, decay=0.999,
                    is_training=is_train, updates_collections=None
                )
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'None':
                _ = _
    return _

def convlstm(input, state, activation=math_ops.sigmoid, kernel_shape=[3, 3],
             norm='batch', is_training=True, reuse=False, name='convlstm'):
    with tf.variable_scope(name, reuse=reuse):
        output_size = input.get_shape().as_list()[-1]
        cell = rnn_cell.ConvLSTMCell(conv_ndims=2,
                                     input_shape=input.get_shape().as_list()[1:],
                                     output_channels=output_size,
                                     kernel_shape=kernel_shape,
                                     skip_connection=False,
                                     initializers=tf.truncated_normal_initializer(stddev=0.02),
                                     activation=activation,
                                     name=name)

        if state is None:
            state = cell.zero_state(input.get_shape().as_list()[0], input.dtype)
        output, new_state = cell(input, state)

        output = bn_act(output, is_training, norm=norm, activation_fn=None)
    return output, new_state

#def resnet_block_convlstm(input, state, activation=None, norm='batch',
#                          is_training=True, reuse=False, name='res_block_convlstm'):
#    with tf.variable_scope(name, reuse=reuse):
#        assert len(state) == 2
#
#        output, state1 = convlstm(input, state[0], activation=math_ops.sigmoid,
#                                  norm=norm,
#                                  is_training=True, reuse=reuse, name=name + '_1')
#        output, state2 = convlstm(output, state[1], activation=math_ops.sigmoid,
#                                  norm=norm,
#                                  is_training=True, reuse=reuse, name=name + '_2')
#
#    output = activation(output + input) if activation else output + input 
#    return output, (state1, state2)
    
def resnet_block_convlstm(input, state, activation=None, norm='batch',
                          is_training=True, reuse=False, name='res_block_convlstm'):
    with tf.variable_scope(name, reuse=reuse): 

        output, state1 = convlstm(input, state, activation=math_ops.sigmoid,
                                  norm=norm,
                                  is_training=True, reuse=reuse, name=name + '_1') 

    output = activation(output + input) if activation else output + input
    return output, state1  

def conv_2d(x, channels, kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, name='conv_0', use_coor=True):
    with tf.variable_scope(name):
        if use_coor:
            coor_x, coor_y = tf.meshgrid(tf.range(x.shape.as_list()[2], dtype=tf.float32)/float(x.shape.as_list()[2]), \
                                         tf.range(x.shape.as_list()[1], dtype=tf.float32) /float(x.shape.as_list()[1]))
            coor_x = tf.reshape(coor_x, (1, coor_x.shape.as_list()[0], coor_x.shape.as_list()[1], 1))
            coor_y = tf.reshape(coor_y, (1, coor_y.shape.as_list()[0], coor_y.shape.as_list()[1], 1))
            coor = tf.concat([coor_x, coor_y], -1)
            coor = tf.tile(coor, [x.shape[0],1,1,1])
            x = tf.concat([x, coor], -1)
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


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0', use_coor=True):
    with tf.variable_scope(scope):
        if use_coor:
            coor_x, coor_y = tf.meshgrid(tf.range(x.shape.as_list()[2], dtype=tf.float32)/float(x.shape.as_list()[2]), \
                                         tf.range(x.shape.as_list()[1], dtype=tf.float32) /float(x.shape.as_list()[1]))
            coor_x = tf.reshape(coor_x, (1, coor_x.shape.as_list()[0], coor_x.shape.as_list()[1], 1))
            coor_y = tf.reshape(coor_y, (1, coor_y.shape.as_list()[0], coor_y.shape.as_list()[1], 1))
            coor = tf.concat([coor_x, coor_y], -1)
            coor = tf.tile(coor, [x.shape[0],1,1,1])
            x = tf.concat([x, coor], -1)
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


def style_mod_GAIN(x, dlatent, use_bias=False, name=None):
    shape = x.shape.as_list()
    with tf.variable_scope(name or 'style_mod_GAIN'):
        style =  tf.layers.dense(dlatent, units=shape[-1] * 2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        # style = tf.reshape(style, [-1, 2, shape[-1]] + [1] * (len(shape) - 2))
        style = tf.reshape(style, [-1, 2, 1, 1, shape[-1]])
        # return x * tf.exp(style[:, 0] + 1) + style[:, 1]
        return (style[:, 0] + 1), style[:, 1]
    
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

 
def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(reference_image - generated_image).mean()
#        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list) 

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

# In the Encoder
def PONO(x, epsilon=1e-5):
    mean, var = tf.nn.moments(x, [3], keep_dims=True) 
    std = tf.sqrt(var + epsilon)
    output = (x - mean) / std
    return output, mean, std

def PONO_cov_arc3(x, cut=0, batch_size=16, epsilon=1e-5):  
    f_shape = x.shape.as_list()
    mean, var = tf.nn.moments(x, [3], keep_dims=True)   
    output = (x - mean) / var
    conv_c = x-mean # n*4*4*128
    conv_1 = tf.reshape(conv_c, (f_shape[0], f_shape[1]*f_shape[2], f_shape[3]))#n*16*128
    var = tf.matmul(conv_1, tf.transpose(conv_1,(0,2,1)))/tf.cast(batch_size-1, tf.float32)#n*128*128
    var = tf.reshape(var, (f_shape[0], f_shape[1], f_shape[2], f_shape[1]*f_shape[2]))
    var = tf.nn.softmax(var, -1) 
    if cut>0:
#        cov = tf.nn.top_k(cov, k=cut).values
        var = var[:,:,:,:cut]
    return output, mean, var, var  
 
def PONO_cov(x, cut=0, batch_size=8, epsilon=1e-5):  
    f_shape = x.shape.as_list()
    mean, var = tf.nn.moments(x, [3], keep_dims=True)   
    output = (x - mean) / var
    conv_c = x-mean # n*4*4*128
    conv_1 = tf.reshape(conv_c, (f_shape[0], f_shape[1]*f_shape[2], f_shape[3]))#n*16*128
    cov = tf.matmul(conv_1, tf.transpose(conv_1,(0,2,1)))/tf.cast(batch_size-1, tf.float32)#n*128*128
    cov = tf.reshape(cov, (f_shape[0], f_shape[1], f_shape[2], f_shape[1]*f_shape[2]))
    cov = tf.nn.softmax(cov, -1) 
    if cut>0:
#        cov = tf.nn.top_k(cov, k=cut).values
        cov = cov[:,:,:,:cut]
    return output, mean, var, cov   
# In the Decoder
# one can call MS(x, mean, std)
# with the mean and std are from a PONO in the encoder
def MS(x, beta, gamma):
    return x * gamma + beta

def MS(x, beta, gamma):
    return x * gamma + beta

from PIL import Image
#views = ['240', '010', '200', '190', '041', '050', '051', '140', '130', '080', '090', '120', '110']
views = ['110', '120', '090', '080', '130', '140', '051', '050', '041', '190', '200', '010', '240']
def read_img(img_path):
    # img_path: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128), Image.ANTIALIAS)
    return img

def read_img13(img_path):
    img9 = []
    for tmp in range(13):
        view = views[tmp]

        token = img_path.split('/')
        name = token[-1]

        token = name.split('_')
        ID = token[0]
        status = token[2]
        bright = token[4]

        img2_path = '/home/ubuntu/ymy233/zzy/CR-GAN-master/data/multi_PIE_crop_128/' + ID + '/' + ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'
        img2 = read_img(img2_path)
        img2 = img2.resize((128, 128), Image.ANTIALIAS)
        img9.append(np.array(img2))
    
    return np.concatenate(img9, 1), np.array(img9)
