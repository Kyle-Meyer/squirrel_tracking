import cv2 
import numpy as np 
import sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
import pickle 

class BoundingBox:
    def __init__(self, aXCoord, aYCoord, aWidth, aHeight, aConfidence: float = 1.0):
        self.mXCoord = aXCoord 
        self.mYCoord = aYCoord
        self.mWidth = aWidth
        self.mHeight = aHeight 
        self.mConfidence = aConfidence

    def center(self):
        return (self.mXCoord + self.mWidth // 2, self.mYCoord + self.mHeight // 2)


class SquirrelDetector:


