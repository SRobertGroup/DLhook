import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.util import img_as_float
from PIL import Image, ImageOps, ImageEnhance




"""
This class preprocess the images which will be passed on as input to the deep learning model
The preprocessing is applied to reduce noise and sharpen the edges of the seedlings
"""
class preprocess_images():
    def __init__(self):
        pass
    def preprocess(self,img_x):
        height, width, _= img_x.shape
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
        im1 = cv2.medianBlur(img_x, 3)
        im1=Image.fromarray(im1)
        enh = ImageEnhance.Contrast(im1)
        im2=enh.enhance(1.2)
        im1=np.array(im2)
        norm_img=np.zeros((height,width))
        final=cv2.normalize(im1, norm_img, 0, 255, cv2.NORM_MINMAX)
        return final
