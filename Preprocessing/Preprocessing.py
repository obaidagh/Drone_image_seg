import numpy as np, pandas as pd, cv2 as cv

import tensorflow as tf
import os
import math

from tensorflow.keras.utils import Sequence

import imgaug as ia
from imgaug import augmenters

from functools import partial

from sklearn.model_selection import train_test_split

#from Extra.RGB_Mask import RGB_to_Mask
from Preprocessing.Augmentaion_Pipeline import  image_augmentation
from Preprocessing.Decode_csv import decode_csv,read_and_decode_orginal,read_and_decode_segmented










def custom_generator(x_set,y_set,reshape_dims,Num_Classes,batch_sizebatch_size):

    class Custom_Sequence(Sequence):

        def __init__(self, x_set, y_set,reshape_dims,Num_Classes,batch_size):
            self.x, self.y    = x_set, y_set
            self.batch_size   = batch_size
            self.reshape_dims = reshape_dims
            self.Num_Classes  = Num_Classes

        
        def __len__(self):
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            return np.array([ read_and_decode_orginal(file_name, reshape_dims) for file_name in batch_x]),\
                   np.array([ read_and_decode_segmented(file_name, reshape_dims,Num_Classes) for file_name in batch_y])

    generator = Custom_Sequence(x_set,y_set,reshape_dims,Num_Classes,batch_sizebatch_size)
    
    return generator













def Ready_data( Orginal_path,Mask_path,Orginal_aug,mask_aug , Key_prop,pipeline,split_percent, Augment=False,Batch_dataset=True):
    #file list
    ORIGINAl_LIST = tf.io.gfile.glob(Orginal_path+'*')
    ORIGINAl_LIST.sort()

    Mask_LIST = tf.io.gfile.glob(Mask_path+'*')
    Mask_LIST.sort()

    #augmentaion


    if Augment:
        image_augmentation(ORIGINAl_LIST,Orginal_aug,pipeline)#orginal augmentaion
        image_augmentation(Mask_LIST,mask_aug,pipeline)#mask augmentaion


    ORIGINAL_JPG_LIST_with_aug = tf.io.gfile.glob(Orginal_aug+'/*')
    ORIGINAL_JPG_LIST_with_aug+=ORIGINAl_LIST
    ORIGINAL_JPG_LIST_with_aug.sort()

    Mask_PNG_LIST_with_aug = tf.io.gfile.glob(mask_aug+'/*')
    Mask_PNG_LIST_with_aug+=Mask_LIST
    Mask_PNG_LIST_with_aug.sort()

    index_with_aug = [os.path.splitext(filename)[0].split('/')[-1] for filename in ORIGINAL_JPG_LIST_with_aug]
    index_with_aug.sort()

    # Creating Dataframe from paths
    df=pd.DataFrame((index_with_aug,ORIGINAL_JPG_LIST_with_aug,Mask_PNG_LIST_with_aug)).T
    df.rename(columns = {
        0: 'index',
        1: 'Orginal',
        2: 'Segmented'},

            inplace=True)

    df.set_index('index', inplace= True)

    X_train,X_val = train_test_split(df, test_size=0.2, random_state=42)
    X_val,X_test = train_test_split(X_val, test_size=0.25, random_state=42)
    
    if Batch_dataset:
        #Convert dataframe to tensorflow BatchDataset
        
        Train_DF= (tf.data.Dataset.from_tensor_slices((X_train['Orginal'].values,X_train['Segmented'].values)))
        Train_BatchDataset = (Train_DF.map(partial(decode_csv, Key_prop["combined"]))).batch(Key_prop["batch_size"]).shuffle(3, reshuffle_each_iteration=True)
        
        Eval_DF = (tf.data.Dataset.from_tensor_slices((X_val['Orginal'].values,X_val['Segmented'].values)))
        Eval_BatchDataset  = (Eval_DF.map(partial(decode_csv, Key_prop["combined"]))).batch(Key_prop["batch_size"]).shuffle(3, reshuffle_each_iteration=True)
        
        return Train_BatchDataset ,Eval_BatchDataset ,X_test
    else:
        train_generator = custom_generator(
            X_train['Orginal'].values,
            X_train['Segmented'].values,
            Key_prop["combined"][1],
            Key_prop["combined"][0],
            Key_prop["batch_size"])

        eval_generator = custom_generator(
            X_train['Orginal'].values,
            X_train['Segmented'].values,
            Key_prop["combined"][1],
            Key_prop["combined"][0],
            Key_prop["batch_size"])
        return train_generator,eval_generator,X_test











