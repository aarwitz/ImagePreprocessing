


import cv2
import os
import shutil
import segment
import numpy as np
from skimage.filters import threshold_otsu

def toBinary(path):
    """
    converts every image to black and white
    """
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, img_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bw

def toSegmented(path):
    img_seg = segment.segment(cv2.imread(path))
    return img_seg

def toOtsuThresholded(path):
    sample_image = cv2.imread(path)
    img = sample_image
   # img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray < thresh
    def filter_image(image, mask):

        r = image[:,:,0] * mask
        g = image[:,:,1] * mask
        b = image[:,:,2] * mask

        return np.dstack([r,g,b])
    filtered = filter_image(img, img_otsu)
    return filtered

def toColorMask(path):
    sample_image = cv2.imread(path)
   # img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
    low = np.array([0, 0, 0])
    high = np.array([215, 85, 86])
    mask = cv2.inRange(sample_image, low, high)
    #result = cv2.bitwise_and(img, img, mask=mask)
    return mask

def toKmeans(path):
    sample_image = cv2.imread(path)
    img = sample_image
  #  img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
    twoDimage = sample_image.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    attempts=10
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

def to2DFilter(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5),np.float32)/25  
    # Applying  Filter2D function to the image
    #img = cv2.filter2D(img,-1,kernel)
    img = cv2.boxFilter(img, 0, (3,3), img, (-1,-1), True, cv2.BORDER_DEFAULT)
    return img

def toMono(path):
    """
    converts every image to mono
    """

def toBrightnessContrast(path,alpha,beta):
    """
    image in path to contrast level alpha and brightness level beta
    """
    image = cv2.imread(path)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def toBlur(path):
    """
    images in path to blur level num
    """
    import cv2
    import numpy

    # read image
    image = cv2.imread(path)

    # apply guassian blur on src image
    dst = cv2.GaussianBlur(image,(253,253),cv2.BORDER_DEFAULT)

    # display input and output image
    return dst

if __name__ == "__main__":
    """"
    Define path and choose visual filter function to run
    """
    img_path = r"/home/aaron/ImagePreprocessing/Data/raw_images/"
    output_path = r"/home/aaron/ImagePreprocessing/Data/output/"

    # clear folder
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # For each file in dataset, apply filter
    for filename in os.listdir(img_path):
        """ 
        toBinary()
        toMono()
        toSegmented()
        ...
        """
        img = toBinary(img_path + filename)
        cv2.imwrite(output_path + filename, img)
        #img2 = toBrightnessContrast(output_path + filename,5,15)
        #cv2.imwrite(output_path2 + filename, img2)
        #img3 = toBW(output_path2 + filename)
        #cv2.imwrite(output_path3 + filename, img3)
    




# Sources
#https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/