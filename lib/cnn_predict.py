########################################################################
##
##  Use run_predict.py to run this script
##  - Predicts labels on a Stargart disease image (image.png)
##
########################################################################

import configparser
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import model_from_json
from lib.helper_functions import count_pixels, convert_to_mm, preds_to_images


## Load parameters from config file
config = configparser.RawConfigParser()
config.read_file(open(r'./config.txt'))
input_dir = config.get('input folder name', 'input_dir')
result_dir = config.get('result folder name', 'result_dir')
train_mean = float(config.get('training settings', 'train_set_mean'))
train_std = float(config.get('training settings', 'train_set_std'))

## Load the model and best weights
model = model_from_json(open('./src/starseg_architecture.json').read())
model.compile()
model.load_weights('./src/starseg_best_weights.h5')

## Load the image and check size and mode
img = Image.open('./' + input_dir + '/image.png')
img_width, img_height = img.size 
assert img_width == 4000 & img_height == 4000, 'Incorrect input image dimensions. A 4000x4000 pixel Optos autofluorescence grayscale image is required.'
img_color = img.mode
if img_color == 'RGB':
    img = img.convert('L')
elif img_color == 'L':
    pass
else: 
    raise AssertionError('Unsupported input image color mode. A 4000x4000 pixel Optos autofluorescence image in RGB or grayscale format is required.')
img.save('./' + result_dir + '/image_copy.png', 'PNG')

## Crop image to macula and downsize
img = img.crop((1744, 1744, 2256, 2256))
img = img.resize((256, 256))
img.save('./' + result_dir + '/image_macula.png', 'PNG')

## Convert image to array and fit to ResNet input layer
img_array = tf.keras.preprocessing.image.img_to_array(img, data_format='channels_last', dtype='float32')
img_array -= train_mean
img_array = img_array / (train_std + tf.keras.backend.epsilon())
img_array = np.repeat(img_array, 3, axis=2)
img_array = np.expand_dims(img_array, axis=0)

## Create and save the predicted label
pred_array = model.predict(img_array, batch_size=1)
pred_array = preds_to_images(pred_array, threshold=0.5)
pred_array = np.squeeze(pred_array)
pred_label = tf.keras.preprocessing.image.array_to_img(pred_array, data_format='channels_last', dtype='float32')
pred_label.save('./' + result_dir + '/predicted_label_macula.png', 'PNG')

## Save calculated total lesion areas in text file
red, green, blue = count_pixels(pred_label)
red = round(convert_to_mm(red), 2)
green = round(convert_to_mm(green), 2)
blue = round(convert_to_mm(blue), 2)
with open('./' + result_dir + '/predicted_total_areas.txt', 'w') as file:
    file.write('Predicted Total Lesion Areas (millimeters-squared):\n\n')
    file.write(f'Definitely Decreased Autofluorescence (DDAF) in Blue: {blue} \n')
    file.write(f'Questionably Decreased Autofluorescence (QDAF) in Green: {green} \n')
    file.write(f'Other Abnormal Autofluorescence (OAAF) in Red: {red} \n')