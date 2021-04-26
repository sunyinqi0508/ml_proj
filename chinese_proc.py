import cv2
import numpy as np
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14,14))
chs = np.fromfile('chinese', dtype='uint8').reshape(62*122, 128, 128)
res = [None for _ in range(0, 62*122)]

def filter(i):
    i = i if i < 10 or i > 100 else i - 10  
    return int((i*i*i)/(255*255))

for i in range(0, 62*122):
    ext = cv2.copyMakeBorder(chs[i], 192,192,192,192,cv2.BORDER_CONSTANT, value=255)
    dilate = cv2.bitwise_not(cv2.dilate(cv2.bitwise_not(ext), kernel, iterations=1))
    resize = cv2.resize(dilate, dsize=(128,128), interpolation=cv2.INTER_NEAREST)
    fil = np.array([filter(int(i)) for i in resize.flatten()]).reshape(128,128).tolist()
    res[i] = fil

res = np.array(res).reshape(62*122, 128, 128).astype('uint8')
print(res.shape)
res.tofile('chinese_proc')