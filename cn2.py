import sys
import os
import cv2
import numpy as np
def tostr(i):
    if(i < 10):
        return '00'+str(i)
    elif (i < 100):
        return '0'+str(i)
    else:
        return str(i)
idxs = [i for i in range(48,58)]
idxs+=[i for i in range(65, 91)]
cn2 = []
cmax = 0
rmax = 0
kk = ['train', 'test']
def filter(i):
    i = int(i)
    i = i if i < 10 or i > 100 else i - 10  
    return int((i*i*i)/(255*255))
labels = np.empty(63714)
n = 0
populate_label = True
for j in range(0, 36):
    cls = idxs[j]
    s_cls = chr(cls)
    k = 0
    ii = 1
    while True:

        img = cv2.imread('./cn2/'+kk[k]+'/'+ s_cls +'/'+str(ii)+'.png')

        ii += 1
        if img is None and k == 1:
            # print (ii)
            break
        elif img is None:
            k += 1
            print (ii)
            ii = 1
            continue
        
        if  j >= 10 and ii % 2 == 1:
            labels[n] = j+26
        else:
            labels[n] = j
        n+=1
        
        if populate_label:
            continue 

        arr = []
        for ir in img:
            c = []
            for ic in ir:
                c.append(filter(ic[0]))
            arr.append(c)
        w = len(c)
        h = len(arr)

        if(w > 128 or h > 128):
            if(w > h):
                w_ = 128
                h_ = (h*128)//w
                padding0 = (128 - h_)//2
                padding1 = 128 - h_ - padding0
                resize = cv2.resize(np.array(arr), dsize=(w_, h_), interpolation=cv2.INTER_NEAREST)
                padding = cv2.copyMakeBorder(resize, padding0, padding1, 0, 0, cv2.BORDER_CONSTANT, value=255)
            else:
                h_ = 128
                w_ = (w*128)//h
                padding0 = (128 - w_)//2
                padding1 = 128 - w_ - padding0
                resize = cv2.resize(np.array(arr), dsize=(w_, h_), interpolation=cv2.INTER_NEAREST)
                padding = cv2.copyMakeBorder(resize, 0,0,padding0, padding1, cv2.BORDER_CONSTANT, value=255)
        else:
            padding0 = (128 - w) // 2
            padding1 = 128 -w- padding0
            padding2 = (128 - h) // 2
            padding3 = 128-h - padding2
            padding = cv2.copyMakeBorder(np.array(arr), padding2, padding3, padding0, padding1, cv2.BORDER_CONSTANT, value=255)
        cn2.append(padding.tolist())
        if(padding.shape[0]!= 128 or padding.shape[1] != 128):
            print(padding.shape)
if(not populate_label):
    ncn2 = np.array(cn2).astype('uint8')
    print(ncn2.shape)        
    ncn2.tofile('cn2.data')

nlabels = np.array(labels).astype('uint8')
print(nlabels.shape)
nlabels.tofile('cn2.label')

