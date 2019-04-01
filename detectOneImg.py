from ctypes import *
from MobileNetSSD import *
import cv2
import time

def main():
#    result = cdll.LoadLibrary("/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/build/lib/liblibmobileNet.so")
    #imgfile = '/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/testPic/test.jpg'
    imgfile = '/home/ubuntu/newdisk/zpx/opencv/uav0000117_02622_v/0001.jpg'
    frame = cv2.imread(imgfile)
    mobilenetssd = MobileNetSSD()
    mobilenetssd.initial()
    a = time.time()
    mobilenetssd.detect(frame)
    b = time.time()
    print(b-a)
    output = mobilenetssd.getRectangle()
    print output[0]
    for i in range(0, int(output[0])):
        print("index")
        print(i)
        x1 = int(output[6*i + 3])
        y1 = int(output[6*i + 4])
        x2 = int(output[6*i + 5])
        y2 = int(output[6*i + 6])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),1)
    cv2.imshow("detect",frame)
    cv2.waitKey(300000)
if __name__ == "__main__":
    main()
