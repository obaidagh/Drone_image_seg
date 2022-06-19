# credit to :https://www.kaggle.com/code/bulentsiyah/preprocessing-rgb-img-masks-to-segmentation-masks

import os,re
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image



#def RGB_to_Mask(Class_dict,RGB_path,Mask_path,limit):
def RGB_to_Mask(Class_dict,RGB_path,Mask_path):


    if not os.path.exists(Mask_path):
        os.mkdir(Mask_path)

    for i,item in enumerate(RGB_path):
       # if i < limit[0] or i > limit[1]:
       #     continue


        root,file = os.path.split(item)
        image_rgb = Image.open(item)
        image_rgb = np.asarray(image_rgb)
        new_image = np.zeros((image_rgb.shape[0],image_rgb.shape[1],3)).astype('int')

        for index, row  in Class_dict.iterrows():
                new_image[(image_rgb[:,:,0]==row.r)&
                            (image_rgb[:,:,1]==row.g)&
                            (image_rgb[:,:,2]==row.b)]=np.array([index+1,index+1,index+1]).reshape(1,3)
        
        new_image = new_image[:,:,0]
        output_path= os.path.join(Mask_path,file)
        cv.imwrite(output_path,new_image)
        del root,file,image_rgb,new_image,output_path
