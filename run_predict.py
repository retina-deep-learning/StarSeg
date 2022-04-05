########################################################################
##
##  Use this file to run the prediction script.
##
########################################################################

import os
import configparser


## Read configuration file
config = configparser.RawConfigParser()
config.read_file(open(r'./config.txt'))
input_dir = config.get('input folder name', 'input_dir')
result_dir = config.get('result folder name', 'result_dir')

## Check for input image file
assert os.path.exists('./' + input_dir + '/image.png') , 'No image.png file was found in the input folder.'

## Create a folder for results
if os.path.exists(result_dir):
    print('Old results folder already exists, please delete or rename the old folder and re-run the script.')
    exit()
else:
    os.makedirs(result_dir)

## Run the prediction script
exec(open('./lib/cnn_predict.py').read())