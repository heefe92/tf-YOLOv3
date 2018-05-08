import Detector
import cv2
import time

def DoTrain(detector):
    detector.Training()

def DoTest(detector):
    start=time.time()
    cv2.imwrite('test2.jpg', detector.Test(imgPath='test.jpg'))
    print(time.time()-start)

if __name__=="__main__":
    detector=Detector.Detector(isRestore=True)
    # DoTrain(detector)
    DoTest(detector)
    # DoTrain(detector)
    # for j in range(1000):
    #     startTime=time.time()
    #     # for i in range(30//10):
    #     DoTest(detector)
    #     print(time.time()-startTime)