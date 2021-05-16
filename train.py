import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/extras/CUPTI/lib64")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras import activations
import cv2
import datetime
from os import listdir


# labels = np.fromfile('labels',dtype='uint32').reshape(62*122)
# y = [0 for i in range(0, 62*122+63714)]
# y = np.array(y + [ 1 for i in range(0, 124000)])

y=np.zeros(62*122+63714+124000, dtype='uint8')
y[62*122+63714:] = 1

cn1 = np.fromfile('chinese_proc', dtype='uint8').reshape(62*122, 128, 128,1)
cn2 = np.fromfile('cn2.data', dtype='uint8').reshape(63714, 128, 128,1)
# chinese = np.concatenate((cn1, cn2), axis=0)
west = np.fromfile('west2k', dtype='uint8').reshape(124000, 128, 128,1)
# data = np.column_stack(chinese, west)
data = np.concatenate((cn1, cn2, west), axis=0)
# data = west
cn1_label = np.fromfile('cn.label', dtype ='uint8')
cn2_label = np.fromfile('cn2.label', dtype='uint8')
west_label = np.fromfile('west.label', dtype='uint8')
labels = np.concatenate((cn1_label, cn2_label, west_label), axis=0)

# labels = west_label
labels = np.where(labels>=36, labels - 36, labels)
train_x, test_x, train_y, test_y = train_test_split(data, y,shuffle=True, train_size=0.9)

train_x.tofile('train_x')
train_y.tofile('train_y')
test_x.tofile('test_x')
test_y.tofile('test_y')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# data = datasets.cifar10.load_data()
def split_west():
    global train_x, test_x, train_y, test_y
    d = west
    l = west_label
    l = np.where(l>=36, l - 36, l)
    train_x, test_x, train_y, test_y = train_test_split(d, l,shuffle=True, train_size=0.9)
    train_x.tofile('train_x')
    train_y.tofile('train_y')
    test_x.tofile('test_x')
    test_y.tofile('test_y')

def split_cn():
    global train_x, test_x, train_y, test_y
    d = np.concatenate((cn1, cn2), axis=0)
    l = np.concatenate((cn1_label, cn2_label), axis=0)
    l = np.where(l>=36, l - 36, l)
    train_x, test_x, train_y, test_y = train_test_split(d, l,shuffle=True, train_size=0.9)
    train_x.tofile('train_x')
    train_y.tofile('train_y')
    test_x.tofile('test_x')
    test_y.tofile('test_y')

def split_all():
    global train_x, test_x, train_y, test_y
    d =  np.concatenate((cn1, cn2, west), axis=0)
    l = np.concatenate((cn1_label, cn2_label, west_label), axis=0)
    l = np.where(l>=36, l - 36, l)
    train_x, test_x, train_y, test_y = train_test_split(d, l,shuffle=True, train_size=0.9)
    train_x.tofile('train_x')
    train_y.tofile('train_y')
    test_x.tofile('test_x')
    test_y.tofile('test_y')

def dnn_categorial_model():
    model = models.Sequential()
    model.add(layers.Conv2D(256, (6, 6), activation='relu', input_shape=(128,128,1)))
    # model.add(layers.Dropout(rate=0.9))
    model.add(layers.MaxPooling2D((6, 6)))
    model.add(layers.Conv2D(256, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (6, 6), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(36)) # one dimenstion per cat

    model.summary()
    adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def dnn_binary_model():
    model = models.Sequential()
    model.add(layers.Conv2D(256, (6, 6), activation='relu', input_shape=(128,128,1)))
    # model.add(layers.Dropout(rate=0.9))
    model.add(layers.MaxPooling2D((6, 6)))
    model.add(layers.Conv2D(256, (6, 6), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (6, 6), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # one dimenstion per cat
    model.summary()
    adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam_optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def dnn_train(model):
    checkpoint_path = "cp4.ckpt"
    logdir="logs/fit/"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    verbose=1)
    _pass = 0
    b = 64
    while (b > 0):
        model.fit(train_x, train_y, epochs=10, batch_size=b,
                        validation_data=(test_x, test_y), 
                        callbacks = [cp_callback,tensorboard_callback])
        b = int(input())
        model.save('pass'+str(_pass))
    return model


def dnn_load_weights(model, file):
    model.load_weights(file)
    return model

def valid_binary(model):
    yhat = model.predict(test_x, batch_size=1)
    print(yhat)
    yhat = [0 if i <= 0 else 1 for i in yhat]
    equality = tf.math.equal(yhat, test_y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    print(accuracy)

def getch(ch):
    ch = ord(ch)
    if (ch) >= ord('a'):
            return 36 + (ch)-ord('a')
    elif (ch) >= ord('A'):
            return 10 + (ch) - ord('A')
    else:
            return (ch) - ord('0')

def y2chr(y):
    if y <  10:
        return chr(y+48)
    elif y < 36:
        return chr(y-10+97)

def bounding_box(img):
    rmin = 129
    rmax = -1
    cmin = 129
    cmax = -1
    i = 0
    for r in img:
        j = 0
        for c in r:
            if(c < 255):
                rmin = rmin if rmin < i else i
                rmax = rmax if rmax > i else i
                cmin = cmin if cmin < j else j
                cmax = cmax if cmax > j else j
            j += 1
        i +=1
    return rmin,cmin, rmax-rmin, cmax-cmin
def knn_preproc(data):
    new_x = np.empty((data.shape[0], 24* 24))
    n = 0
    for img in train_x:
        ho, wo, h, w = bounding_box(img)
        if h > w:
            diff = h - w
            diff_2 = diff // 2
            diff -= diff_2
            if (wo >= diff_2):
                wo -= diff_2
            else:
                wo = 0
                diff += diff_2 - wo
            if(w + wo + diff <= 128):
                w += diff
            else:
                w = 128
        elif (h < w):
            diff = w - h
            diff_2 = diff//2
            diff -= diff_2
            if (ho >= diff_2):
                ho -= diff_2
            else:
                ho = 0
                diff += diff_2 - wo
            if(h + ho + diff <= 128):
                h += diff
            else:
                h = 128
        crop = tf.image.crop_to_bounding_box(img, ho, wo, h, w).numpy()
        resize = cv2.resize(crop, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('a', resize)
        # cv2.waitKey(0)
        new_x[n] = resize.reshape(24*24)
        return new_x.astype('uint8')

def knn(data, labels, test, tlabels):
    classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
    classifier.fit(data,labels)
    print(np.count_nonzero(classifier.predict(test)))
    print(tlabels.shape)
    score = classifier.score(test, tlabels)
    print(score)
    return classifier
def knn_cv (data, labels):
    classifier = cv2.ml.KNearest_create()
    classifier.train(data, cv2.ml.ROW_SAMPLE, labels)
    return classifier

def from_categorial(y):
    max = float('-inf')
    n = 0
    i = 0
    for yy in y:
        if yy > max:
            max = yy
            n = i
        i+=1
    return n

def train_accent_recognizer():
    split_all()
    binary_model = dnn_binary_model()
    binary_model = dnn_train(binary_model)
    valid_binary(binary_model)
    return binary_model

def train_western_ocr():
    split_west()
    cat_model = dnn_categorial_model()
    cat_model = dnn_train(cat_model)
    return cat_model

def train_cn_ocr ():
    split_cn()
    cat_model = dnn_categorial_model()
    cat_model = dnn_train()
    return cat_model

def train_all_ocr() :
    split_all()
    cat_model = dnn_categorial_model()
    cat_model = dnn_train()
    return cat_model

rec = train_accent_recognizer()
wocr = train_western_ocr()
cocr = train_cn_ocr()
aocr = train_all_ocr()

imgs = [cv2.imread(dir+img, cv2.IMREAD_GRAYSCALE) for img in listdir(dir)]
eug = np.array(imgs).reshape(len(imgs),128,128,1)
yhatall = aocr.predict(eug)
yhatsel = [cocr.predict(e) if rec.predict(e)<0 else wocr.predict(e) for e in eug]

clsall = [y2chr(from_categorial(ymat)) for ymat in yhatall]
print(clsall)
clssel = [y2chr(from_categorial(ymat)) for ymat in yhatsel]
print(clssel)

#binary_model = dnn_binary_model()
# binary_model = dnn_load_weights(binary_model, './.9994/cp4.ckpt')
#binary_model = dnn_train(binary_model)
# valid_binary(binary_model)
# split_west()
# cat_model_w = dnn_categorial_model()
# cat_model_w = dnn_load_weights(cat_model_w, '../variables')
# cat_model_w = dnn_train(cat_model_w)
# #dir = 'X:/Users/bill/Desktop/ML/proj/realworldDS/Eugene/'
# dir = './Eugene/'
# imgs = [cv2.imread(dir+img, cv2.IMREAD_GRAYSCALE) for img in listdir(dir)]
# eug = np.array(imgs).reshape(len(imgs),128,128,1)
# yhat = cat_model_w.predict(eug)

# cls = [y2chr(from_categorial(ymat)) for ymat in yhat]
# print(cls)
# classifier = knn(knn_preproc(train_x), train_y, knn_preproc(test_x), test_y )
# knn_yhat = classifier.predict(knn_preproc(eug))
# print(knn_yhat)
# classifier = knn_cv(knn_preproc(train_x), train_y)
