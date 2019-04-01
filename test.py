from ctypes import *
from MobileNetSSD import *
import cv2
import time
import numpy as np

def main():
#    result = cdll.LoadLibrary("/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/build/lib/liblibmobileNet.so")
    mobilenetssd = MobileNetSSD()
#    mobilenetssd.initial()
   
#    src = np.random.randint(1,100,size=(2,2,3))
    src_ = cv2.imread('/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/testPic/test.jpg')
    src = np.asarray(src_, dtype=np.int)
#    src = np.asarray(src)
    src = src.ctypes.data_as(c_char_p)
    
    row = src_.shape[0]
    col = src_.shape[1]
    channels = src_.shape[2]
    print row, col, channels
    mobilenetssd.loadimg(src, row, col, channels)
if __name__ == "__main__":
    main()
