import os
import numpy as np
from PIL import Image
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters



def image_augmentation(image_path,augmentaion_path,pipeline):
    globed=tf.io.gfile.glob(image_path)

    if not os.path.exists(augmentaion_path):
        os.mkdir(augmentaion_path)

    for item in globed:
        

        root,file = os.path.split(item)
        h=os.path.splitext(file)


        image = Image.open(item)
        image = np.array(image)


        ia.seed(1)
        for i,step in enumerate(pipeline):
            temp = step.augment_image(image)

            output_path= os.path.join(augmentaion_path,h[0]+"_"+str(i)+h[1])

            im = Image.fromarray(temp)
            im.save(output_path)
            del im,temp,output_path


