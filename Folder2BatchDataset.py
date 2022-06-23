import numpy as np, pandas as pd, cv2 as cv

import tensorflow as tf
import os
from PIL import Image

import imgaug as ia
from imgaug import augmenters

from functools import partial

from sklearn.model_selection import train_test_split

from RGB_Mask import RGB_to_Mask
from Augmentaion_Pipeline import  image_augmentation
from Decode_csv import decode_csv






def TO_batch(Orginal_path,Mask_path,Key_prop,pipeline,split_percent,Augment=False):
    #file list
    ORIGINAl_LIST = tf.io.gfile.glob(Orginal_path+'*')
    ORIGINAl_LIST.sort()

    Mask_LIST = tf.io.gfile.glob(Mask_path+'*')
    Mask_LIST.sort()

    #augmentaion


    if Augment:
        image_augmentation(ORIGINAl_LIST,Orginal_path,pipeline)#orginal augmentaion
        image_augmentation(Mask_LIST,Mask_path,pipeline)#mask augmentaion


    ORIGINAL_JPG_LIST_with_aug = tf.io.gfile.glob(Orginal_path+'*')
    ORIGINAL_JPG_LIST_with_aug.sort()

    Mask_PNG_LIST_with_aug = tf.io.gfile.glob(Mask_path+'*')
    Mask_PNG_LIST_with_aug.sort()

    index_with_aug = [os.path.splitext(filename)[0] for filename in os.listdir(Orginal_path)]
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
    

    #Convert dataframe to tensorflow BatchDataset
    
    Train_DF= (tf.data.Dataset.from_tensor_slices((X_train['Orginal'].values,X_train['Segmented'].values)))
    
    Train_BatchDataset = (Train_DF.map(partial(decode_csv, Key_prop["combined"]))).batch(Key_prop["batch_size"]).shuffle(3, reshuffle_each_iteration=True)

    Eval_DF = (tf.data.Dataset.from_tensor_slices((X_val['Orginal'].values,X_val['Segmented'].values)))
    
    Eval_BatchDataset  = (Eval_DF.map(partial(decode_csv, Key_prop["combined"]))).batch(Key_prop["batch_size"]).shuffle(3, reshuffle_each_iteration=True)
    
    return Train_BatchDataset ,Eval_BatchDataset ,X_test,df
