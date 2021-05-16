import numpy as np
#import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt
# import sklearn as skl
import cv2
def read_chinese():
    chinese = [[] for _ in range(0, 62)]

    for k in range(1, 123):
        file = open('./character/'+ str(k)+ '_images', 'rb')
        nchars = int.from_bytes(file.read(4), 'little')
        w = int.from_bytes(file.read(1), 'little')
        h = int.from_bytes(file.read(1), 'little')
        charlen = w*h
        chars = []
        i = 0
        while(i < 62):
            j = 0
            curr_char = []
            while(j < charlen):
                curr_char.append(int.from_bytes(file.read(1), 'little'))
                j += 1
            chinese[i].append(curr_char)
            i += 1
    n_chinese = np.array(chinese).astype('uint8')
    n_chinese.tofile('chinese')


def read_west_l(name):
    west = np.empty((124000*128*128),dtype='uint8')
    def tostr(i):
        if(i < 10):
            return '000'+str(i)
        elif (i < 100):
            return '00'+str(i)
        elif(i < 1000):
            return '0' + str(i)
        else:
            return str(i)
    idxs = [hex(i)[2:] for i in range(48,58)]
    idxs+=[hex(i)[2:] for i in range(65, 91)]
    idxs+= [hex(i)[2:] for i in range(97, 123)]
    n = 0
    for j in range(0, 62):
        cls = idxs[j]
        s_cls = str(cls)
        k = 0
        ii = 0
        print(j)
        for i in range(0, 2000):
            img = cv2.imread('X:/Users/bill/Desktop/ML/proj/code/by_class/'+ s_cls +'/hsf_'+str(k)+'/hsf_'+str(k)+'_0'+tostr(ii)+'.png')
            while img is None:
                ii = 0
                k += 1
                print('k='+str(k))
                img = cv2.imread('X:/Users/bill/Desktop/ML/proj/code/by_class/'+ s_cls +'/hsf_'+str(k)+'/hsf_'+str(k)+'_0'+tostr(ii)+'.png')
            ii += 1
            for ir in img:
                for ic in ir:
                    west[n]=(ic[0])
                    n+=1
            
    #print(len(chars))
    
    n_west = np.array(west).astype('uint8')
    print(n_west.shape)
    n_west.tofile(name)

def write_labels():
    idxs = [hex(i)[2:] for i in range(48,58)]
    idxs+=[hex(i)[2:] for i in range(65, 91)]
    idxs+= [hex(i)[2:] for i in range(97, 123)]
    labels = [[chr(int(j, 16)) for _ in range(0, 122)] for j in idxs]
    n_labels = np.array(labels).flatten()
    n_labels.tofile('labels')

def write_labels2():
    cn2order=[i for i in range(0,62)]
    westorder = [i for i in range(0, 10)] 
    westorder += [i for i in range(36, 62)]
    westorder += [i for i in range(10, 36)]

    cnlabels = [j//122 for j in range(0, 122*62)]
    west_labels=[(westorder[j//2000]) for j in range(0, 2000*62)]
    np.array(cnlabels).astype('uint8').tofile('cn.label')
    np.array(west_labels).astype('uint8').tofile('west.label')
# read_west_l('west2k')

write_labels2()