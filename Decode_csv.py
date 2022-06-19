import tensorflow as tf


def read_and_decode_orginal(filename, reshape_dims):
    
    # 1. Read the file.
    img = tf.io.read_file(filename)
    # 2. Convert the compressed string to a 3D float32 tensor.
    img = tf.io.decode_image( img, channels=3,dtype=tf.dtypes.float32,expand_animations = False)
    # 3. Resize the image to the desired size.

    return tf.image.resize(img, reshape_dims)

def read_and_decode_segmented(filename, reshape_dims,Num_Classes):

    # 1. Read the file.
    img = tf.io.read_file(filename)
    # 2. Convert the compressed string to a 3D int tensor.
    img = tf.io.decode_image( img,channels=1,dtype=tf.dtypes.int32,expand_animations = False)
    
    img = tf.image.resize(img, reshape_dims)
    img = tf.reshape(img,reshape_dims)

    img = tf.image.convert_image_dtype(img, dtype=tf.dtypes.int32)
    # 3. Resize the image to the desired size.
    img = tf.one_hot(img, Num_Classes, dtype=tf.int32)
    
    return img

def decode_csv(combined,orginal_path,segmented_path):
    orginal_img   = read_and_decode_orginal(orginal_path, combined[1])
    segmented_img = read_and_decode_segmented(segmented_path, combined[1],combined[0])
    return orginal_img, segmented_img