
from cv2 import *
import math
from component_tree import  *


import scipy.stats as st


#gaussian -> drivitive of gaussianned pic


"""""

#building gaussian kernel and derive it
def derive_gauss():
    retval = cv2.getGaussianKernel( 3, 1 )
    gauss_ker = np.dot( retval, retval.reshape( 1, 3 ) )



get_gauss_kernel = getGaussianKernel(9, 5, CV_32F)
#ans = mulTransposed(get_gauss_kernel,get_gauss_kernel,False)
#print(ans)

"""
def gauss_scale_space(img):
    ans = []
    #read the image
    #convert to graysclae
    gray_image = cvtColor( img, COLOR_BGR2GRAY )

    #get rows and columns size
    img_size = gray_image.shape
    rows = img_size[0]
    cols = img_size[1]

    #binarize and flip
    thresh =127
    print(type(gray_image))
    ret_val2,im_bw = threshold( gray_image, thresh, 255,THRESH_BINARY )
    waitKey(0)
    im_wb = bitwise_not(im_bw)


    scale_space = [None]*100

    #creating rows*cols array for result
    ans_pic_min = np.zeros(dtype=np.float32,shape=(rows,cols))
    ans_pic_max = np.zeros(dtype=np.float32,shape=(rows,cols))
    ans_pic_min_int8 = ans_pic_min.astype(np.int8)
    ans_pic_max_int8 = ans_pic_max.astype(np.int8)
    s=0


    #generating guassined smoothed pictures (creating the scale space)
    for i in range(11,37,2):
        scale_space[s] = GaussianBlur( im_wb, (i*9, i), 0 )
       # print( "kernel hight:", i, ",index:", s )
        s=s+1

    #loop each pixel
    #for eache pixel check for strongest response (inner loop)
    for x in range(0, rows):
        for y in range(0, cols):
            min = math.inf
            max = -math.inf
            for i in range(0,s):
                mat=scale_space[i]
                pixel = mat[x,y]
                if (pixel<min):
                    ans_pic_min_int8[x,y] = pixel
                    min=pixel
                if (pixel > max):
                    ans_pic_max_int8[x,y] = pixel
                    max=pixel

    ans.append(ans_pic_min_int8)
    ans.append(ans_pic_max_int8)


     #retval, labels = connectedComponents(gray_image)
    retval, labels = connectedComponents(image=ans_pic_min_int8,connectivity=8,ltype=CV_32S)
    ans.append(labels)

    return ans
def mark_pic(components,mat):
    ans = mat
    for node in components:
        for x,y in node.data:
           ans[x,y] = 100
    return ans

