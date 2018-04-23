import Detector
import cv2
import time

def DoTrain(detector):
    detector.Training()

def DoTest(detector):
    start=time.time()
    cv2.imshow('test', detector.Test(imgPath='Z:/dataset/KNU-Campus Dataset/images/20180312_172240/20180312_172240_0005.jpg'))
    print(time.time()-start)
    cv2.waitKey(0)

if __name__=="__main__":
    detector=Detector.Detector(isRestore=True)
    DoTrain(detector)
    DoTest(detector)
    # DoTrain(detector)
    # for j in range(1000):
    #     startTime=time.time()
    #     # for i in range(30//10):
    #     DoTest(detector)
    #     print(time.time()-startTime)