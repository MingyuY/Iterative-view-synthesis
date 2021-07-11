#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:30:09 2018

@author: crazydemo
"""

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

image_height=64
image_width=64

def data2fig_1c(samples, img_size=64, nr=4, nc=4):
        #if self.is_tanh:
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_size, img_size), cmap='gray')
        return fig

def data2fig(samples, img_size=64, nr=4, nc=4):
        #if self.is_tanh:
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_size, img_size, 3), cmap='Greys_r')
        return fig

def data2fig_4(samples, img_size=64, nr=4, nc=4):
        #if self.is_tanh:
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(img_size, img_size, 4) )
        return fig
def data2fig_less(samples, img_size=64, nr=2, nc=2):
    # if self.is_tanh:
    samples = (samples + 1) / 2
    fig = plt.figure(figsize=(2, 2))
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size, 3), cmap='Greys_r')
    return fig

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [178, 178, 3])
    img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label

def read_and_decode_lsun(filename):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'img_raw' : tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img

def read_and_decode_flower_noc(filename):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'img_raw' : tf.FixedLenFeature([], tf.string)})
    # features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
    #                                                                 'img_raw' : tf.FixedLenFeature([], tf.string)})
    # label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    # return img,label
    return img

def read_and_decode_flower(filename, size=64, is_train=True):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(serialized_example,features={'img_raw' : tf.FixedLenFeature([], tf.string)})
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    if is_train:
        img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label
    # return img
    
def read_and_decode_mnist(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'id': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    idlabel = tf.cast(features['id'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 1])
    img = tf.cast(img, tf.float32)
    return img,idlabel
    
def read_and_decode_multi_PIE(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'id': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    idlabel = tf.cast(features['id'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label,idlabel
    # return img 
    
def read_and_decode_multi_PIE_matrix(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.VarLenFeature(tf.float32),
                                                                    'id': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.sparse_tensor_to_dense(features['label'], default_value=0)
#    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [7])
    idlabel = tf.cast(features['id'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label,idlabel

def read_and_decode_shapenet_chair(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)                    
    features = tf.parse_single_example(serialized_example,features={'elevation': tf.FixedLenFeature([], tf.float32),
                                                                    'azimuth': tf.FixedLenFeature([], tf.float32),
                                                                    'distance': tf.FixedLenFeature([], tf.float32),
                                                                    'label': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    elevation = tf.cast(features['elevation'], tf.float32)
    azimuth = tf.cast(features['azimuth'], tf.float32)
    distance = tf.cast(features['distance'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [137, 137, 4])
#    img = img[:,:,:3]
    # img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [size, size])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32) 
#    return img, elevation, azimuth, distance
    return img, label, elevation, azimuth, distance

def read_and_decode_chair(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'poselabel': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    poselabel = tf.cast(features['poselabel'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label,poselabel    
def read_and_decode_chair_new(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),
                                                                    'p': tf.FixedLenFeature([], tf.int64),
                                                                    't': tf.FixedLenFeature([], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    p = tf.cast(features['p'], tf.int32)
    t = tf.cast(features['t'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label, p, t  
def read_and_decode_chair_paired_new(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'posea': tf.FixedLenFeature([], tf.int64),
                                                                    'pa': tf.FixedLenFeature([], tf.int64),
                                                                    'ta': tf.FixedLenFeature([], tf.int64),
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'poseb': tf.FixedLenFeature([], tf.int64),
                                                                    'pb': tf.FixedLenFeature([], tf.int64),
                                                                    'tb': tf.FixedLenFeature([], tf.int64),
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string)})
    posea = tf.cast(features['posea'], tf.int32)
    pa = tf.cast(features['pa'], tf.int32)
    ta = tf.cast(features['ta'], tf.int32)
    imga = tf.decode_raw(features['img_rawa'], tf.uint8)
    imga = tf.reshape(imga, [size, size, 3])
    imga = tf.cast(imga, tf.float32)
    
    poseb = tf.cast(features['poseb'], tf.int32)
    pb = tf.cast(features['pb'], tf.int32)
    tb = tf.cast(features['tb'], tf.int32)
    imgb = tf.decode_raw(features['img_rawb'], tf.uint8)
    imgb = tf.reshape(imgb, [size, size, 3])
    imgb = tf.cast(imgb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return imga, posea, pa, ta, imgb, poseb, pb, tb
    # return img

def read_and_decode_multi_PIE_paired_matrix(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={'expressiona': tf.FixedLenFeature([], tf.int64),
                                                                    'ida': tf.FixedLenFeature([], tf.int64),
                                                                    'azimutha': tf.VarLenFeature(tf.float32),
                                                                    'lightnessa': tf.FixedLenFeature([], tf.int64), 
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'expressionb': tf.FixedLenFeature([], tf.int64),
                                                                    'idb': tf.FixedLenFeature([], tf.int64),
                                                                    'azimuthb': tf.VarLenFeature(tf.float32),
                                                                    'lightnessb': tf.FixedLenFeature([], tf.int64), 
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    })
    expressiona = tf.cast(features['expressiona'], tf.int32)
    ida = tf.cast(features['ida'], tf.int32)
    azimutha = tf.sparse_tensor_to_dense(features['azimutha'], default_value=0)
#    azimutha = tf.decode_raw(features['azimutha'], tf.float32)
    azimutha = tf.reshape(azimutha, [7])
    lightnessa = tf.cast(features['lightnessa'], tf.int32)
    img_rawa = tf.decode_raw(features['img_rawa'], tf.uint8)
    img_rawa = tf.reshape(img_rawa, [size, size, 3])
    img_rawa = tf.cast(img_rawa, tf.float32)
    expressionb = tf.cast(features['expressionb'], tf.int32)
    idb = tf.cast(features['idb'], tf.int32)
    azimuthb = tf.sparse_tensor_to_dense(features['azimuthb'], default_value=0)
#    azimuthb = tf.decode_raw(features['azimuthb'], tf.float32)
    azimuthb = tf.reshape(azimuthb, [7])
    lightnessb = tf.cast(features['lightnessb'], tf.int32)
    img_rawb = tf.decode_raw(features['img_rawb'], tf.uint8)
    img_rawb = tf.reshape(img_rawb, [size, size, 3])
    img_rawb = tf.cast(img_rawb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return img_rawa, ida, azimutha, img_rawb, idb, azimuthb

def read_and_decode_300w_LP_paired(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={'ida': tf.FixedLenFeature([], tf.string),
                                                                    'azimutha': tf.FixedLenFeature([], tf.int64),
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'idb': tf.FixedLenFeature([], tf.string),
                                                                    'azimuthb': tf.FixedLenFeature([], tf.int64),
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    }) 
    ida = tf.cast(features['ida'], tf.string)
    azimutha = tf.cast(features['azimutha'], tf.int32) 
    img_rawa = tf.decode_raw(features['img_rawa'], tf.uint8)
    img_rawa = tf.reshape(img_rawa, [size, size, 3])
    img_rawa = tf.cast(img_rawa, tf.float32) 
    idb = tf.cast(features['idb'], tf.string)
    azimuthb = tf.cast(features['azimuthb'], tf.int32)
    img_rawb = tf.decode_raw(features['img_rawb'], tf.uint8)
    img_rawb = tf.reshape(img_rawb, [size, size, 3])
    img_rawb = tf.cast(img_rawb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return img_rawa, ida, azimutha, img_rawb, idb, azimuthb

def read_and_decode_multi_PIE_paired(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={'expressiona': tf.FixedLenFeature([], tf.int64),
                                                                    'ida': tf.FixedLenFeature([], tf.int64),
                                                                    'azimutha': tf.FixedLenFeature([], tf.int64),
                                                                    'lightnessa': tf.FixedLenFeature([], tf.int64), 
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'expressionb': tf.FixedLenFeature([], tf.int64),
                                                                    'idb': tf.FixedLenFeature([], tf.int64),
                                                                    'azimuthb': tf.FixedLenFeature([], tf.int64),
                                                                    'lightnessb': tf.FixedLenFeature([], tf.int64), 
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    })
    expressiona = tf.cast(features['expressiona'], tf.int32)
    ida = tf.cast(features['ida'], tf.int32)
    azimutha = tf.cast(features['azimutha'], tf.int32)
    lightnessa = tf.cast(features['lightnessa'], tf.int32)
    img_rawa = tf.decode_raw(features['img_rawa'], tf.uint8)
    img_rawa = tf.reshape(img_rawa, [size, size, 3])
    img_rawa = tf.cast(img_rawa, tf.float32)
    expressionb = tf.cast(features['expressionb'], tf.int32)
    idb = tf.cast(features['idb'], tf.int32)
    azimuthb = tf.cast(features['azimuthb'], tf.int32)
    lightnessb = tf.cast(features['lightnessb'], tf.int32)
    img_rawb = tf.decode_raw(features['img_rawb'], tf.uint8)
    img_rawb = tf.reshape(img_rawb, [size, size, 3])
    img_rawb = tf.cast(img_rawb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return img_rawa, ida, azimutha, img_rawb, idb, azimuthb


def read_and_decode_fasion_paired1(filename, size, sixe_w=0):
    if sixe_w!=0:
        sizew = sixe_w
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={'parsing_rawa': tf.FixedLenFeature([], tf.string), 
                                                                    'pose_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'parsing_rawb': tf.FixedLenFeature([], tf.string), 
                                                                    'pose_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    }) 
    img_rawa = tf.decode_raw(features['img_rawa'], tf.uint8)
    img_rawa = tf.reshape(img_rawa, [size, sizew, 3])
    img_rawa = tf.cast(img_rawa, tf.float32)
    img_rawb = tf.decode_raw(features['img_rawb'], tf.uint8)
    img_rawb = tf.reshape(img_rawb, [size, sizew, 3])
    img_rawb = tf.cast(img_rawb, tf.float32)
    
    parsing_rawa = tf.decode_raw(features['parsing_rawa'], tf.uint8)
    parsing_rawa = tf.reshape(parsing_rawa, [size, sizew, 3])
    parsing_rawa = tf.cast(parsing_rawa, tf.float32)
    parsing_rawb = tf.decode_raw(features['parsing_rawb'], tf.uint8)
    parsing_rawb = tf.reshape(parsing_rawb, [size, sizew, 3])
    parsing_rawb = tf.cast(parsing_rawb, tf.float32)
    
    pose_rawa = tf.decode_raw(features['pose_rawa'], tf.int64)
    pose_rawa = tf.reshape(pose_rawa, [18, 2])
    pose_rawa = tf.cast(pose_rawa, tf.float32)
    pose_rawb = tf.decode_raw(features['pose_rawb'], tf.int64)
    pose_rawb = tf.reshape(pose_rawb, [18, 2])
    pose_rawb = tf.cast(pose_rawb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return img_rawa, parsing_rawa, pose_rawa, img_rawb, parsing_rawb, pose_rawb

def read_and_decode_fasion_paired(filename, size):
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,features={'parsing_rawa': tf.FixedLenFeature([], tf.string), 
                                                                    'img_rawa' : tf.FixedLenFeature([], tf.string),
                                                                    'parsing_rawb': tf.FixedLenFeature([], tf.string), 
                                                                    'img_rawb' : tf.FixedLenFeature([], tf.string),
                                                                    }) 
    img_rawa = tf.decode_raw(features['img_rawa'], tf.uint8)
    img_rawa = tf.reshape(img_rawa, [size, size, 3])
    img_rawa = tf.cast(img_rawa, tf.float32)
    img_rawb = tf.decode_raw(features['img_rawb'], tf.uint8)
    img_rawb = tf.reshape(img_rawb, [size, size, 3])
    img_rawb = tf.cast(img_rawb, tf.float32)
    
    parsing_rawa = tf.decode_raw(features['parsing_rawa'], tf.uint8)
    parsing_rawa = tf.reshape(parsing_rawa, [size, size, 3])
    parsing_rawa = tf.cast(parsing_rawa, tf.float32)
    parsing_rawb = tf.decode_raw(features['parsing_rawb'], tf.uint8)
    parsing_rawb = tf.reshape(parsing_rawb, [size, size, 3])
    parsing_rawb = tf.cast(parsing_rawb, tf.float32)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    return img_rawa, parsing_rawa, img_rawb, parsing_rawb

def read_and_decode_test(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [178, 178, 3])
    img = tf.image.resize_images(img, [image_height, image_width])
    #img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label

def read_and_decode_test_ordinal(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([40], tf.int64),
                                                                    'img_raw' : tf.FixedLenFeature([], tf.string),
                                                                    'name': tf.FixedLenFeature([], tf.int64)})
    name = tf.cast(features['name'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.image.resize_image_with_crop_or_pad(img, image_height, image_width)
#    img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    return img,label,name


#imga,  idlabela,  labela,  imgb,  idlabelb,  labelb = read_and_decode_multi_PIE_paired(['/home/ymy/zzy/datasets/multi_PIE_train_paired.tfrecords'], 128)
#img_batcha, label_batcha, img_batchb, label_batchb = tf.train.shuffle_batch([imga, idlabela, imga, idlabelb, ],
#                                                                  batch_size=10,
#                                                                  capacity=1000, min_after_dequeue=600)
#from tqdm import tqdm
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(sess=sess)
#for i in tqdm(range(10000)):
#    X_batch_a, Y_batch_a, X_batch_b, Y_batch_b = sess.run([ img_batcha, label_batcha, img_batchb, label_batchb])
#    if Y_batch_a.any()==120 or Y_batch_a.any()==191:
#        print 1
