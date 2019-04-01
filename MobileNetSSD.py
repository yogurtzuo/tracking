from ctypes import *
import numpy as np

class MobileNetSSD:            

    def __init__(self):        
        self.ldmobileNetSSD = cdll.LoadLibrary('/home/ubuntu/newdisk/mobile-ssd/MobileNet-SSD-TensorRT/build/lib/liblibmobileNet.so')                      
        self.mobileNetssd = self.ldmobileNetSSD.new_mobileNetSSD()
             
    def __del__(self):                                             
        self.ldmobileNetSSD.destorySSD(self.mobileNetssd)          

    def initial(self):                                             
        self.ldmobileNetSSD.initMobileNetSSD(self.mobileNetssd)

    def detect(self, frame):
        matrix = np.asarray(frame, dtype = np.int)
        matrix = matrix.ctypes.data_as(c_char_p)
        rows = frame.shape[0]
        cols = frame.shape[1]
        channels = frame.shape[2]
        print rows, cols, channels
        self.ldmobileNetSSD.inferDetect(self.mobileNetssd, matrix, rows, cols, channels)

    def getRectangle(self):                                    
        self.ldmobileNetSSD.getOutput.restype = POINTER(c_float)
        return self.ldmobileNetSSD.getOutput(self.mobileNetssd)
   
#    def loadimg(self, matrix, rows, cols, channels):
#        self.ldmobileNetSSD.load_img(self.mobileNetssd, matrix, rows, cols, channels)
#
