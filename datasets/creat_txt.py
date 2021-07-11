#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:41:44 2019

@author: ymy
"""
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from itertools import permutations
import linecache
from glob import glob
import pandas as pd
import pose_transform
import pose_utils


pi = 3.1416 # 180 degree
d_60 = pi / 3
d_15 = pi / 12
d_range = pi / 36 # 5 degree

d_45 = d_60 - d_15
d_30 = d_45 - d_15

def get_300w_LP_img(img_path):
    # img_path = '/home/ubuntu/ymy233/zzy/CR-GAN-master/data/crop_0822/AFW_resize/AFW_1051618982_1_0_128.jpg'
    #/home/ubuntu/ymy233/zzy/CR-GAN-master/data/crop_0907/AFW_resize/AFW_134212_1_0_128.jpg
    # txt_path: /home/ubuntu/ymy233/zzy/CR-GAN-master/data/300w_LP_size_128/AFW_resize/AFW_1051618982_1_0_128_pose_shape_expression_128.txt
    right = img_path.find('_128.jpg')
    for i in range(right-1, 0, -1):
        if img_path[i] == '_':
            left = i
            break
    
    view2 = -1
    while(view2 < 0):
        tmp = random.randint(0, 17)
        new_txt = img_path[:left+1] + str(tmp) + '_128_pose_shape_expression_128.txt'
        new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")
        
        if os.path.isfile(new_txt):
            param = np.loadtxt(new_txt)
            yaw = param[1]
            if yaw < -d_60 or yaw > d_60:
                view2 = -1
            elif yaw >= -d_60 and yaw < -d_60+d_range:# -60 -55 = -60
                view2 = 0
            elif yaw >= -d_45-d_range and yaw < -d_45+d_range:#-50 -40 = -45
                view2 = 1
            elif yaw >= -d_30-d_range and yaw < -d_30+d_range:#-35 -25 = -30
                view2 = 2
            elif yaw >= -d_15-d_range and yaw < -d_15+d_range:#-20 -10 = -15
                view2 = 3
            elif yaw >= -d_range and yaw < d_range:#-5 5 = 0
                view2 = 4
            elif yaw >= d_15-d_range and yaw < d_15+d_range:#10 - 20 = 15
                view2 = 5
            elif yaw >= d_30-d_range and yaw < d_30+d_range:#25 -35 = 30
                view2 = 6
            elif yaw >= d_45-d_range and yaw < d_45+d_range:#40 -50 = 45
                view2 = 7
            elif yaw >= d_60-d_range and yaw <= d_60:#55 60 = 60
                view2 = 8
    
    new_img = img_path[:left+1] + str(tmp) + '_128.jpg'
    img2 = read_img( new_img )
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    
    return view2, img2
        
def creat_300w_LP_paired_tf(path):
    import os
    img_dir = path
    from tqdm import tqdm
    with open('300w_LP_train_paired.txt','w') as f:
        img_list = glob(img_dir+'/*/' + '*_*_0_128.jpg' )
        for i in tqdm(range(len(img_list))):
            #/home/ymy/ymy/data/crop_0907/AFW_resize/AFW_1139324862_1_0_128.jpg
            image_name = img_list[i]
            pose_loc = image_name.find('_128.jpg')-1
            image_name_one = image_name[:pose_loc]+'*'+ image_name[pose_loc+1:]
            one_limg_list = glob(image_name_one)
            one_limg_list_ = []
            for kk in range(len(one_limg_list)):
                 if np.sum(one_limg_list[kk][14:]==np.array(list2)):
                     one_limg_list_.append(one_limg_list[kk])
            pairs = list(zip(*list(permutations(one_limg_list_, 2))))
            np.random.seed(10) 
#            if i==212:
#                print 1
            if len(pairs)>0:
                perm1 = np.random.permutation(pairs[0])[:20]
                perm2 = np.random.permutation(pairs[1])[:20]  
                for A,B in zip(perm1, perm2):
                    new_txt = A[:pose_loc+1] + '_128_pose_shape_expression_128.txt'
                    new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")
                    if os.path.isfile(new_txt):
                        param = np.loadtxt(new_txt)
                        yaw = param[1]
                        if yaw < -d_60 or yaw > d_60:
                            viewA = -1
                        elif yaw >= -d_60 and yaw < -d_60+d_range:# -60 -55 = -60
                            viewA = 0
                        elif yaw >= -d_45-d_range and yaw < -d_45+d_range:#-50 -40 = -45
                            viewA = 1
                        elif yaw >= -d_30-d_range and yaw < -d_30+d_range:#-35 -25 = -30
                            viewA = 2
                        elif yaw >= -d_15-d_range and yaw < -d_15+d_range:#-20 -10 = -15
                            viewA = 3
                        elif yaw >= -d_range and yaw < d_range:#-5 5 = 0
                            viewA = 4
                        elif yaw >= d_15-d_range and yaw < d_15+d_range:#10 - 20 = 15
                            viewA = 5
                        elif yaw >= d_30-d_range and yaw < d_30+d_range:#25 -35 = 30
                            viewA = 6
                        elif yaw >= d_45-d_range and yaw < d_45+d_range:#40 -50 = 45
                            viewA = 7
                        elif yaw >= d_60-d_range and yaw <= d_60:#55 60 = 60
                            viewA = 8
                    id_A = A.split('/')[-1].split('_')[0]+'_'+ A.split('/')[-1].split('_')[1]
                     
                    new_txt = B[:pose_loc+1] + '_128_pose_shape_expression_128.txt'
                    new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")
                    if os.path.isfile(new_txt):
                        param = np.loadtxt(new_txt)
                        yaw = param[1]
                        if yaw < -d_60 or yaw > d_60:
                            viewB = -1
                        elif yaw >= -d_60 and yaw < -d_60+d_range:# -60 -55 = -60
                            viewB = 0
                        elif yaw >= -d_45-d_range and yaw < -d_45+d_range:#-50 -40 = -45
                            viewB = 1
                        elif yaw >= -d_30-d_range and yaw < -d_30+d_range:#-35 -25 = -30
                            viewB = 2
                        elif yaw >= -d_15-d_range and yaw < -d_15+d_range:#-20 -10 = -15
                            viewB = 3
                        elif yaw >= -d_range and yaw < d_range:#-5 5 = 0
                            viewB = 4
                        elif yaw >= d_15-d_range and yaw < d_15+d_range:#10 - 20 = 15
                            viewB = 5
                        elif yaw >= d_30-d_range and yaw < d_30+d_range:#25 -35 = 30
                            viewB = 6
                        elif yaw >= d_45-d_range and yaw < d_45+d_range:#40 -50 = 45
                            viewB = 7
                        elif yaw >= d_60-d_range and yaw <= d_60:#55 60 = 60
                            viewB = 8
                    if not(viewA<0 or viewB<0):
                        id_A = A.split('/')[-1].split('_')[0]+'_'+ A.split('/')[-1].split('_')[1]
                        id_B = B.split('/')[-1].split('_')[0]+'_'+ B.split('/')[-1].split('_')[1]
                        f.write(A[19:])
                        f.write(' ')
                        f.write(B[19:])
                        f.write(' ')
                        f.write(str(viewA))
                        f.write(' ')
                        f.write(str(viewB))
                        f.write(' ')
                        f.write(id_A)
                        f.write(' ')
                        f.write(id_B)
                        f.write(' ')
                        f.write('\n')
        f.close()

        
def creat_multiPIE_paired_tf(path):
    img_dir = path
    order = np.arange(1, 250)
#    np.random.shuffle(order)
    delect_index = np.where(order==213)
#    test_index1 = np.where(order==120)
#    test_index2 = np.where(order==191)
    order = np.delete(order, delect_index)
#    order = np.delete(order, test_index1)
#    order = np.delete(order, test_index2)
#    train_num = order[:200]
#    test_num = np.array(list(order[200:]) + [120, 191])
    train_num = order[:200]
    test_num = np.array(list(order[200:212]) + list(order[213:]))
    #img_list = glob(img_dir+'/*/*.png') 
#    views = {240:0, 200:1, 190:2, 140:3, 130:4, 120:5, 110:6, 90:7, 80:8, 51:9, 50:10, 41:11, 10:12}
    views = {110:0, 120:1, 90:2, 80:3, 130:4, 140:5, 51:6, 50:7, 41:8, 190:9, 200:10, 10:11, 240:12}
    train_num = np.load('/home/ymy/zzy/file-others/multiPIE_id/train_id.npy')
    test_num = np.load('/home/ymy/zzy/file-others/multiPIE_id/test_id.npy')
    with open('multiPIE_train_paired.txt','w') as f:
    #    for i in range(10):
        for _, i in enumerate(train_num):
            for j in range(20):
                for k in range(1,3,1):
                    img_list = glob(img_dir+'/' + str(i).zfill(3) + '/*_01_' + str(k).zfill(2) + '_*_'+ str(j).zfill(2) +'_crop_128.png' ) 
                    pairs = list(zip(*list(permutations(img_list, 2))))
                    np.random.seed(10) 
                    perm1 = np.random.permutation(pairs[0])[:20]
                    perm2 = np.random.permutation(pairs[1])[:20]  
                    for A,B in zip(perm1, perm2):
                        str_a = A.split('/')[-2] + '/' + A.split('/')[-1]
                        str_b = B.split('/')[-2] + '/' + B.split('/')[-1]
                        f.write(str_a)
                        f.write(' ')
                        f.write(str_b)
                        f.write(' ')
                        name = A.split('/')[-1]#101_01_01_120_00_crop_128.png
                        f.write(name.split('_')[0])#id
                        f.write(' ') 
                        f.write(name.split('_')[2])#expression
                        f.write(' ') 
                        f.write(str(views[int(name.split('_')[3])])) #azimuth
                        f.write(' ')  
                        f.write(name.split('_')[4])#lightness
                        f.write(' ') 
                        name = B.split('/')[-1]
                        f.write(name.split('_')[0])#id
                        f.write(' ') 
                        f.write(name.split('_')[2])#expression
                        f.write(' ') 
                        f.write(str(views[int(name.split('_')[3])])) #azimuth
                        f.write(' ')  
                        f.write(name.split('_')[4])#lightness 
                        f.write('\n')
        f.close()
    
    with open('multiPIE_test_paired.txt','w') as f:
    #    for i in range(10):
        for _, i in enumerate(test_num):
            for j in range(20):
                for k in range(1,3,1):
                    img_list = glob(img_dir+'/' + str(i).zfill(3) + '/*_01_' + str(k).zfill(2) + '_*_'+ str(j).zfill(2) +'_crop_128.png' ) 
                    pairs = list(zip(*list(permutations(img_list, 2))))
                    np.random.seed(10) 
                    perm1 = np.random.permutation(pairs[0])[:20]
                    perm2 = np.random.permutation(pairs[1])[:20]  
                    for A,B in zip(perm1, perm2):
                        str_a = A.split('/')[-2] + '/' + A.split('/')[-1]
                        str_b = B.split('/')[-2] + '/' + B.split('/')[-1]
                        f.write(str_a)
                        f.write(' ')
                        f.write(str_b)
                        f.write(' ')
                        name = A.split('/')[-1]
                        f.write(name.split('_')[0])#id
                        f.write(' ') 
                        f.write(name.split('_')[2])#expression
                        f.write(' ') 
                        f.write(str(views[int(name.split('_')[3])])) #azimuth
                        f.write(' ')  
                        f.write(name.split('_')[4])#lightness
                        f.write(' ') 
                        name = B.split('/')[-1]
                        f.write(name.split('_')[0])#id
                        f.write(' ') 
                        f.write(name.split('_')[2])#expression
                        f.write(' ') 
                        f.write(str(views[int(name.split('_')[3])])) #azimuth
                        f.write(' ')  
                        f.write(name.split('_')[4])#lightness 
                        f.write('\n')
        f.close()

        
 
def creat_chair_paired_tf(path):
    img_dir = path
    info_dic =  path+'/all_chair_names.mat'
    import scipy.io as scio 
    info_mat = scio.loadmat(info_dic) 
    folder_names = info_mat['folder_names'][0]
    order = np.arange(len(folder_names))
#    np.random.shuffle(order)
    train_num = order[:int(len(folder_names)*0.8)+1]
    test_num = order[int(len(folder_names)*0.8)+1:]  
    train_name = folder_names[train_num]
    test_name = folder_names[test_num]
    with open('chair_train_paired.txt','w') as f:
    #    for i in range(10):
        for _, i in enumerate(train_name):
            img_list = glob(img_dir+'/' + str(i[0]) + '/*/*.png')
            pairs = list(zip(*list(permutations(img_list, 2))))
            np.random.seed(10)
            perm1 = np.random.permutation(pairs[0])[:20]
            perm2 = np.random.permutation(pairs[1])[:20]  
            for A,B in zip(perm1, perm2):
                str_a = A.split('/')[-3] + '/' + A.split('/')[-2] + '/' + A.split('/')[-1]
                str_b = B.split('/')[-3] + '/' + B.split('/')[-2] + '/' + B.split('/')[-1]
                f.write(str_a)
                f.write(' ')
                f.write(str_b)
                f.write(' ')
                name = A.split('/')[-1]
                f.write(name.split('_')[1])
                f.write(' ') 
                f.write(name.split('_')[2])
                f.write(' ') 
                f.write(name.split('_')[3])
                f.write(' ') 
                name = B.split('/')[-1]
                f.write(name.split('_')[1])
                f.write(' ') 
                f.write(name.split('_')[2])
                f.write(' ') 
                f.write(name.split('_')[3])
                f.write('\n')
        f.close()
    
    with open('chair_test_paired.txt','w') as f:
    #    for i in range(10):
        for _, i in enumerate(test_name):
            img_list = glob(img_dir+'/' + str(i[0]) + '/*/*.png')
            pairs = list(zip(*list(permutations(img_list, 2))))
            np.random.seed(10) 
            perm1 = np.random.permutation(pairs[0])[:20]
            perm2 = np.random.permutation(pairs[1])[:20]  
            for A,B in zip(perm1, perm2):
                str_a = A.split('/')[-3] + '/' + A.split('/')[-2] + '/' + A.split('/')[-1]
                str_b = B.split('/')[-3] + '/' + B.split('/')[-2] + '/' + B.split('/')[-1]
                f.write(str_a)
                f.write(' ')
                f.write(str_b)
                f.write(' ')
                name = A.split('/')[-1]
                f.write(name.split('_')[1])
                f.write(' ') 
                f.write(name.split('_')[2])
                f.write(' ') 
                f.write(name.split('_')[3])
                f.write(' ') 
                name = B.split('/')[-1]
                f.write(name.split('_')[1])
                f.write(' ') 
                f.write(name.split('_')[2])
                f.write(' ') 
                f.write(name.split('_')[3])
                f.write('\n')
        f.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_multiPIE', type=str, default=None, help='datasets path')
    parser.add_argument('--path_chair', type=str, default=None, help='datasets path')
    parser.add_argument('--path_300w_LP', type=str, default=None, help='datasets path')
    args = parser.parse_args()
    creat_paired_multiPIE_tf(args.path_chair)
    creat_chair_paired_tf(args.path_300w_LP)
    creat_paired_300w_LP_tf(args.path_multiPIE)      
