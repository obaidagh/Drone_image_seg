import tensorflow as tf  
import os 
import numpy as np
from PIL import Image

def resize(filename,channels,reshape_dims):
    img = tf.io.read_file(filename)
    if channels==1:
        img = tf.io.decode_image( img,channels=1,dtype=tf.dtypes.uint8,expand_animations = False)
        img = tf.image.resize(img, reshape_dims,method='nearest')
        img = tf.reshape(img,reshape_dims)
        img = np.array(img)
        im = Image.fromarray(img )




    else:
        img = tf.io.decode_image( img, channels=3,dtype=tf.dtypes.float32,expand_animations = False)
        img = tf.image.resize(img, reshape_dims,method='nearest')
        img = np.array(img)*255
        img = img.astype(np.uint8)
        im = Image.fromarray(img, "RGB")


      






    root,file = os.path.split(filename)

    output_path= os.path.join(root+"/resized/"+file)

    im.save(output_path)


reshape_dims=(300,400)

Orginal_path = "../input/semantic-drone-dataset/dataset1/semantic_drone_dataset/original_images/"
globed_orginal=tf.io.gfile.glob(Orginal_path+"*")
globed_orginal.sort()
globed_orginal = globed_orginal[0:-1]



Mask_path    = "../input/semantic-drone-dataset/dataset1/semantic_drone_dataset/label_images_semantic/"
globed_mask=tf.io.gfile.glob(Mask_path+"*")
globed_mask.sort()
globed_mask = globed_mask[0:-1]


channels= 3

#for item in globed_orginal:
#    output_path=resize(item,channels,reshape_dims)
#    print(output_path)



channels= 1

for item in globed_mask:
    output_path=resize(item,channels,reshape_dims)
    print(output_path)



