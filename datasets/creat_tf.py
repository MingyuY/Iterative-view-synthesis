#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:20:39 2018

@author: ivy
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import time
from glob import glob
import pandas as pd

    
def creat_paired_300w_LP_tf(path):
    list_file  ="../300w_LP_train_paired_shuf.txt"
    root = path
    
    count = 0
    writer = tf.python_io.TFRecordWriter("../300w_LP_train_paired_shuf.tfrecords")
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            field = line.split(' ')
    #        temp = field[1:41]
    #        label=[np.int(i) for i in temp]
            imga = Image.open(root+field[0])
            imgb = Image.open(root+field[1])
            ida = [field[4]] 
            azimutha = [int(field[2])] 
            idb = [field[5]] 
            azimuthb = [int(field[3])]
            if float(imga.size[0])/float(imga.size[1])>4 or float(imga.size[1])/float(imga.size[0])>4:
                continue
    #        img = img.crop((0, 20, 178, 198))  #original align, crop to 178*178; new align, no crop
            # img = img.crop((25, 45, 153, 173))
    #        img= img.resize((64,64))
    #         img.save('../CelebA/Img/'+str(count)+'_128.jpg')
    #         count +=1
            img_rawa = imga.tobytes()
            img_rawb = imgb.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={'ida': tf.train.Feature(bytes_list=tf.train.BytesList(value=ida)),\
                                                                           'azimutha': tf.train.Feature(int64_list=tf.train.Int64List(value=azimutha)),\
                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
                                                                           'idb': tf.train.Feature(bytes_list=tf.train.BytesList(value=idb)),\
                                                                           'azimuthb': tf.train.Feature(int64_list=tf.train.Int64List(value=azimuthb)),\
                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
            # print example
            writer.write(example.SerializeToString())
            count = count + 1
    # #        if count%100000 ==0:break
            if count%500 ==0:
                #print "%d images are processed." %count
                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
        print "%d images are processed." %count
    print 'Done!'
    writer.close()
    
def creat_paired_multiPIE_tf(path):
    list_file  ="../multiPIE_train_paired_shuf.txt"
    list_file_test  ="../multiPIE_test_paired.txt"
    root = path
    
    count = 0
    writer = tf.python_io.TFRecordWriter("../datasets/multi_PIE_train_paired.tfrecords")
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            field = line.split(' ')
    #        temp = field[1:41]
    #        label=[np.int(i) for i in temp]
            imga = Image.open(root+field[0])
            imgb = Image.open(root+field[1])
            ida = [int(field[2])]
            expressiona = [int(field[3])]
            azimutha = [int(field[4])]
            lightnessa = [int(field[5])]
            idb = [int(field[6])]
            expressionb = [int(field[7])]
            azimuthb = [int(field[8])]
            lightnessb = [int(field[9])]
            if float(imga.size[0])/float(imga.size[1])>4 or float(imga.size[1])/float(imga.size[0])>4:
                continue
    #        img = img.crop((0, 20, 178, 198))  #original align, crop to 178*178; new align, no crop
            # img = img.crop((25, 45, 153, 173))
    #        img= img.resize((64,64))
    #         img.save('../CelebA/Img/'+str(count)+'_128.jpg')
    #         count +=1
            img_rawa = imga.tobytes()
            img_rawb = imgb.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={'expressiona': tf.train.Feature(int64_list=tf.train.Int64List(value=expressiona)),\
                                                                           'ida': tf.train.Feature(int64_list=tf.train.Int64List(value=ida)),\
                                                                           'azimutha': tf.train.Feature(int64_list=tf.train.Int64List(value=azimutha)),\
                                                                           'lightnessa': tf.train.Feature(int64_list=tf.train.Int64List(value=lightnessa)),\
                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
                                                                           'expressionb': tf.train.Feature(int64_list=tf.train.Int64List(value=expressionb)),\
                                                                           'idb': tf.train.Feature(int64_list=tf.train.Int64List(value=idb)),\
                                                                           'azimuthb': tf.train.Feature(int64_list=tf.train.Int64List(value=azimuthb)),\
                                                                           'lightnessb': tf.train.Feature(int64_list=tf.train.Int64List(value=lightnessb)),\
                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
            # print example
            writer.write(example.SerializeToString())
            count = count + 1
    # #        if count%100000 ==0:break
            if count%500 ==0:
                #print "%d images are processed." %count
                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
        print "%d images are processed." %count
    print 'Done!'
    writer.close()
    
    count = 0
    writer = tf.python_io.TFRecordWriter("../datasets/multi_PIE_test_paired.tfrecords")
    with open(list_file_test, 'r') as f:
        for line in f:
            line = line.strip()
            field = line.split(' ')
    #        temp = field[1:41]
    #        label=[np.int(i) for i in temp]
            imga = Image.open(root+field[0])
            imgb = Image.open(root+field[1])
            ida = [int(field[2])]
            expressiona = [int(field[3])]
            azimutha = [int(field[4])]
            lightnessa = [int(field[5])]
            idb = [int(field[6])]
            expressionb = [int(field[7])]
            azimuthb = [int(field[8])]
            lightnessb = [int(field[9])]
            if float(imga.size[0])/float(imga.size[1])>4 or float(imga.size[1])/float(imga.size[0])>4:
                continue
    #        img = img.crop((0, 20, 178, 198))  #original align, crop to 178*178; new align, no crop
            # img = img.crop((25, 45, 153, 173))
    #        img= img.resize((64,64))
    #         img.save('../CelebA/Img/'+str(count)+'_128.jpg')
    #         count +=1
            img_rawa = imga.tobytes()
            img_rawb = imgb.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={'expressiona': tf.train.Feature(int64_list=tf.train.Int64List(value=expressiona)),\
                                                                           'ida': tf.train.Feature(int64_list=tf.train.Int64List(value=ida)),\
                                                                           'azimutha': tf.train.Feature(int64_list=tf.train.Int64List(value=azimutha)),\
                                                                           'lightnessa': tf.train.Feature(int64_list=tf.train.Int64List(value=lightnessa)),\
                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
                                                                           'expressionb': tf.train.Feature(int64_list=tf.train.Int64List(value=expressionb)),\
                                                                           'idb': tf.train.Feature(int64_list=tf.train.Int64List(value=idb)),\
                                                                           'azimuthb': tf.train.Feature(int64_list=tf.train.Int64List(value=azimuthb)),\
                                                                           'lightnessb': tf.train.Feature(int64_list=tf.train.Int64List(value=lightnessb)),\
                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
            # print example
            writer.write(example.SerializeToString())
            count = count + 1
    # #        if count%100000 ==0:break
            if count%500 ==0:
                #print "%d images are processed." %count
                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
        print "%d images are processed." %count
    print 'Done!'
    writer.close()

def creat_paired_chair_tf():
    list_file  ="../chair_train_paired_shuf.txt"
    list_file_test  ="../chair_test_paired.txt"
    root = path
    
    count = 0
    writer = tf.python_io.TFRecordWriter("../chair128_train_paired.tfrecords")
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            field = line.split(' ')
    #        temp = field[1:41]
    #        label=[np.int(i) for i in temp]
            imga = Image.open(root+field[0])
            imgb = Image.open(root+field[1])
            posea = [int(field[2])]
            pa = [int(field[3][1:])]
            ta = [int(field[4][1:])]
            poseb = [int(field[5])]
            pb = [int(field[6][1:])]
            tb = [int(field[7][1:])]
            if float(imga.size[0])/float(imga.size[1])>4 or float(imga.size[1])/float(imga.size[0])>4:
                continue 
            imga=imga.crop([120, 120, 480, 480])
            imgb=imgb.crop([120, 120, 480, 480])
            imga= imga.resize((128,128))
            imgb= imgb.resize((128,128))
            img_rawa = imga.tobytes()
            img_rawb = imgb.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={'posea': tf.train.Feature(int64_list=tf.train.Int64List(value=posea)),\
                                                                           'pa': tf.train.Feature(int64_list=tf.train.Int64List(value=pa)),\
                                                                           'ta': tf.train.Feature(int64_list=tf.train.Int64List(value=ta)),\
                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
                                                                           'poseb': tf.train.Feature(int64_list=tf.train.Int64List(value=poseb)),\
                                                                           'pb': tf.train.Feature(int64_list=tf.train.Int64List(value=pb)),\
                                                                           'tb': tf.train.Feature(int64_list=tf.train.Int64List(value=tb)),\
                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
    
#                        tf.train.Example(features=tf.train.Features(feature={'posea': tf.train.Feature(int64_list=tf.train.Int64List(value=posea)),\
#                                                                           'pa': tf.train.Feature(int64_list=tf.train.Int64List(value=pa)),\
#                                                                           'ta': tf.train.Feature(int64_list=tf.train.Int64List(value=ta)),\
#                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
#                                                                           'poseb': tf.train.Feature(int64_list=tf.train.Int64List(value=poseb)),\
#                                                                           'pb': tf.train.Feature(int64_list=tf.train.Int64List(value=pb)),\
#                                                                           'tb': tf.train.Feature(int64_list=tf.train.Int64List(value=tb)),\ 
#                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
            # print example
            writer.write(example.SerializeToString())
            count = count + 1
    # #        if count%100000 ==0:break
            if count%500 ==0:
                #print "%d images are processed." %count
                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
        print "%d images are processed." %count
    print 'Done!'
    writer.close()
    
#    count = 0
#    writer = tf.python_io.TFRecordWriter("../chair128_test_paired.tfrecords")
#    with open(list_file_test, 'r') as f:
#        for line in f:
#            line = line.strip()
#            field = line.split(' ')
#    #        temp = field[1:41]
#    #        label=[np.int(i) for i in temp]
#            imga = Image.open(root+field[0])
#            imgb = Image.open(root+field[1])
#            posea = [int(field[2])]
#            pa = [int(field[3][1:])]
#            ta = [int(field[4][1:])]
#            poseb = [int(field[5])]
#            pb = [int(field[6][1:])]
#            tb = [int(field[7][1:])]
#            if float(imga.size[0])/float(imga.size[1])>4 or float(imga.size[1])/float(imga.size[0])>4:
#                continue
#    #        img = img.crop((0, 20, 178, 198))  #original align, crop to 178*178; new align, no crop
#            # img = img.crop((25, 45, 153, 173))
#    #        img= img.resize((64,64))
#    #         img.save('../CelebA/Img/'+str(count)+'_128.jpg')
#    #         count +=1
#            imga=imga.crop([120, 120, 480, 480])
#            imgb=imgb.crop([120, 120, 480, 480])
#            imga.save('test.jpg')
#            imga= imga.resize((128,128))
#            imgb= imgb.resize((128,128))
#            img_rawa = imga.tobytes()
#            img_rawb = imgb.tobytes()
#            example = tf.train.Example(features=tf.train.Features(feature={'posea': tf.train.Feature(int64_list=tf.train.Int64List(value=posea)),\
#                                                                           'pa': tf.train.Feature(int64_list=tf.train.Int64List(value=pa)),\
#                                                                           'ta': tf.train.Feature(int64_list=tf.train.Int64List(value=ta)),\
#                                                                           'img_rawa': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawa])),\
#                                                                           'poseb': tf.train.Feature(int64_list=tf.train.Int64List(value=poseb)),\
#                                                                           'pb': tf.train.Feature(int64_list=tf.train.Int64List(value=pb)),\
#                                                                           'tb': tf.train.Feature(int64_list=tf.train.Int64List(value=pb)),\
#                                                                           'img_rawb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawb]))}))
#            # print example
#            writer.write(example.SerializeToString())
#            count = count + 1
#    # #        if count%100000 ==0:break
#            if count%500 ==0:
#                #print "%d images are processed." %count
#                print 'Time:{0},{1} images are processed.'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),count)
#        print "%d images are processed." %count
#    print 'Done!'
#    writer.close()


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_multiPIE', type=str, default=None, help='datasets path')
    parser.add_argument('--path_chair', type=str, default=None, help='datasets path')
    parser.add_argument('--path_300w_LP', type=str, default=None, help='datasets path')
    args = parser.parse_args()
    creat_paired_multiPIE_tf(args.path_chair)
    creat_paired_chair_tf(args.path_path_300w_LP)
    creat_paired_300w_LP_tf(args.path_multiPIE)
