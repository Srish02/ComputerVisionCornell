import math
import sys
from sys import argv as args
import cv2
import numpy as np
from numpy.linalg import inv
import scipy.ndimage.filters as scp

EXTRA_CREDIT = False

class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position

def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN

    height, width, _ = img.shape
    h_r = height-1
    w_r = width-1
    
    upper_left = np.array((0, 0, 1))
    upper_right = np.array((0, h_r, 1))
    down_right = np.array((w_r, h_r, 1))
    down_left = np.array((w_r, 0, 1))

    uL = np.dot(M,upper_left)/np.dot(M,upper_left)[2]
    uR = np.dot(M,upper_right)/np.dot(M,upper_right)[2]
    dR = np.dot(M,down_right)/np.dot(M,down_right)[2]
    dL = np.dot(M,down_left)/np.dot(M,down_left)[2]

    x_list = [uL[0],uR[0],dR[0],dL[0]]
    y_list = [uL[1],uR[1],dR[1],dL[1]]

    minX = min(x_list)
    maxX = max(x_list)
    minY = min(y_list)
    maxY = max(y_list)

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)

for arg in args:  
    if arg.find('--extra-credit' ) != -1: 
        EXTRA_CREDIT = True
    else:
        EXTRA_CREDIT = False

def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    img_h = img.shape[0]
    img_w = img.shape[1]

    new_img = np.ones((img_h,img_w),dtype='float32')

    h = acc.shape[1]
    w = acc.shape[0]
    z = 3

    temp = blendWidth/2.0
    extra_crdit_img = np.ones((img_h,img_w),dtype='float32')

    if  EXTRA_CREDIT == True:
        extra_crdit_img = extraCreditBlendAccumulateImage(img, extra_crdit_img, temp)      
        new_img = extra_crdit_img
    
    if EXTRA_CREDIT == False:
        temp_new = blendWidth+1
        new_img[:,:temp_new]=np.linspace(0,1,temp_new)
        temp_new = temp_new
        new_img[:,-blendWidth-1:]=np.linspace(1,0,blendWidth+2-1)
        
    if EXTRA_CREDIT == False:
        new_img = blendAccumulateImage(img, new_img,blendWidth)

    FLAG_TYPE = cv2.INTER_NEAREST
    blend_img = cv2.warpPerspective(new_img, M, (h, w),flags=FLAG_TYPE)
    #blend
    acc[:,:,z] = acc[:,:,z]+blend_img
    for i in range(0,z,1):
        k_t = 0
        h_ = h
        w_ = w
        acc[:,:,i] +=+blend_img *cv2.warpPerspective(img[:,:,i], M, (h, w),
        flags=FLAG_TYPE)
    
    #TODO-BLOCK-END
    # END TODO

def blendAccumulateImage(img, new_img,blendWidth):
    temp = new_img.copy()
    temp_blend = blendWidth+1
    temp[:temp_blend,:]=np.linspace(0,1,temp_blend).reshape(temp_blend,1)
    new_img = np.minimum(temp,new_img)
    new_img[(img[:,:,0]+img[:,:,1]+img[:,:,2])==0]=0
    return new_img

def extraCreditBlendAccumulateImage(img, extra_crdit_img, temp):
    x = img[:,:,0]
    y = img[:,:,1]
    z = img[:,:,2]
    extra_crdit_img[(x+y+z)==0]=-2+1
    extra_crdit_img = scp.gaussian_filter(extra_crdit_img,temp,mode='constant',cval=-1)
    extra_crdit_img = extraCreditMax(extra_crdit_img)
    return extra_crdit_img

def extraCreditMax(extra_crdit_img):
    temp = 0
    extra_crdit_img = np.maximum(extra_crdit_img,0)
    return extra_crdit_img

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    
    height, width, _ = acc.shape
    img = np.zeros((height,width,4),dtype=np.uint8)
    img = calcImageBlend(height, width, img, acc)

    #TODO-BLOCK-END
    # END TODO
    return img

def calcImageBlend(height, width, img, acc):
    for x in range(height):
        for y in range(width):
            for z in range(0,3,1):
                numerator = float(acc[x,y,z])
                denom = float(acc[x,y,3])
                if (acc[x,y,3] != 0):
                    img[x,y,z] = int(numerator/denom)
                else:
                    img[x,y,z] = 0  
            img[x,y,3] = 1 
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        
        min_tX, min_tY, max_tX, max_tY = imageBoundingBox(img, M)
        minX = min(minX, min_tX)
        minY = min(minY, min_tY)
        maxX = max(maxX, max_tX)
        maxY = max(maxY, max_tY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

