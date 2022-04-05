########################################################################
##
##  Use this file to run the training and cross-validation script
##
########################################################################

import os, shutil
import configparser
import tensorflow as tf
from tensorflow.python.framework.config import set_memory_growth


## Read configuration file
config = configparser.RawConfigParser()
config.read_file(open(r'./config.txt'))
result_dir = config.get('result folder name', 'result_dir')

## Create a folder for results

if os.path.exists(result_dir):
    print('Old results folder already exists, please delete or rename the old folder and re-run the script.')
    exit()
else:
    os.makedirs(result_dir)
    shutil.copy('config.txt', result_dir)

    ## Set memory growth for Tensorflow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
## Run the training script
exec(open('./lib/cnn_train.py').read())
