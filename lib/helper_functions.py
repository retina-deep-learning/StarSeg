########################################################################
##
##  Helper functions
##
########################################################################

from sys import builtin_module_names
import h5py
import numpy as np
from keras import backend as K
from PIL import Image


def load_hdf5(data_file_name):
    
    '''
    Loads images and labels from an hdf5 file

    Args:
        data_file_name (string): name of file
        
    Returns:
        images (array): images in shape
                (number of images, height, width, number of channels)
        labels (array): labels in shape
                (number of labels, height, width, number of one-hot encodings)
    '''
    
    images, labels = [], []
    
    file = h5py.File(data_file_name, 'r+')

    images = np.array(file['/images'])
    labels = np.array(file['/labels'])

    return images, labels


def count_pixels(img):

    '''
    Counts the number of pixels of each color

    Args:
        img (object): PIL image object
    
    Returns:
        red, green, blue (tuple of ints): number of red, green, and blue pixels
    '''

    red = 0
    green = 0
    blue = 0
    for pixel in img.getdata():
        if pixel == (255,0,0):
            red += 1
        elif pixel == (0,255,0):
            green += 1
        elif pixel == (0,0,255):
            blue += 1
    
    return int(red), int(green), int(blue)


def convert_to_mm(number):

    '''
    Converts pixels to millimeters-squared
    
    Args:
        number (int): number of pixels
    
    Returns:
        mm (float): number of millimeters
    '''
    
    mm = float(number) * 0.001294745
    
    return mm


def soft_dice_loss(y_true, y_pred, axis=(0,1), epsilon=0.00001):
    
    '''
    Computes the mean soft dice loss over all classes

    Args:
        y_true (tensor): ground truth values for class labels
                        shape: (height, width, number of classes)
        y_pred (tensor): predictions for all class labels
                        shape: (height, width, number of classes)
        axis (tuple): axes (height, width) to sum over when computing numerator and
                      denominator for dice loss
        epsilon (float): tiny constant added to numerator and denominator to
                        avoid divide by 0 errors
                        
    Returns:
        soft_dice_loss (float): computed value of soft dice loss     
    '''
    
    dice_numerator = 2 * (K.sum(y_true*y_pred, axis=axis)) + epsilon
    dice_denominator = K.sum(y_pred**2, axis=axis) + K.sum(y_true**2, axis=axis) + epsilon
    
    soft_dice_loss = 1 - (K.mean(dice_numerator/dice_denominator, axis=0))

    return soft_dice_loss


def dice_coefficient(y_true, y_pred, axis=(0,1), epsilon=0.00001):
    
    '''
    Computes the mean dice coefficient over all classes

    Args:
        y_true (tensor): ground truth values for class labels
                        shape: (height, width, number of classes)
        y_pred (tensor): predictions for all class labels
                        shape: (height, width, number of classes)
        axis (tuple): axes (height, width) to sum over when computing numerator and
                      denominator for dice loss
        epsilon (float): tiny constant added to numerator and denominator to
                        avoid divide by 0 errors

    Returns:
        dice_coefficient (float): computed value of dice coefficient     
    '''
    
    dice_numerator = 2 * (K.sum(y_true*y_pred, axis=axis)) + epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
    
    dice_coefficient = K.mean(dice_numerator/dice_denominator, axis=0)

    return dice_coefficient


def dice_subclass1(y_true, y_pred, axis=(0,1), epsilon=0.00001):
    
    '''
    Computes the dice coefficient for subclass 1 (blue label, DDAF)

    Args:
        y_true (tensor): ground truth values for class labels
                        shape: (height, width, number of classes)
        y_pred (tensor): predictions for all class labels
                        shape: (height, width, number of classes)
        axis (tuple): axes (height, width) to sum over when computing numerator and
                      denominator for dice loss
        epsilon (float): tiny constant added to numerator and denominator to
                        avoid divide by 0 errors
    Returns:
        dice_subclass1 (float): computed value of dice coefficient
                                for DDAF class  
    '''
    
    dice_numerator = 2 * (K.sum(y_true*y_pred, axis=axis)) + epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
    
    dice_subclass1 = (dice_numerator/dice_denominator)[:,0]

    return dice_subclass1


def dice_subclass2(y_true, y_pred, axis=(0,1), epsilon=0.00001):
    
    '''
    Computes the dice coefficient for subclass 2 (green label, QDAF)

    Args:
        y_true (tensor): ground truth values for class labels
                        shape: (height, width, number of classes)
        y_pred (tensor): predictions for all class labels
                        shape: (height, width, number of classes)
        axis (tuple): axes (height, width) to sum over when computing numerator and
                      denominator for dice loss
        epsilon (float): tiny constant added to numerator and denominator to
                        avoid divide by 0 errors

    Returns:
        dice_subclass2 (float): computed value of dice coefficient
                                for QDAF class  
    '''
    
    dice_numerator = 2 * (K.sum(y_true*y_pred, axis=axis)) + epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon

    dice_subclass2 = (dice_numerator/dice_denominator)[:,1]
    
    return dice_subclass2


def dice_subclass3(y_true, y_pred, axis=(0,1), epsilon=0.00001):
    
    '''
    Computes the dice coefficient for subclass 3 (red label, OAAF)

    Args:
        y_true (tensor): ground truth values for class labels
                        shape: (height, width, number of classes)
        y_pred (tensor): predictions for all class labels
                        shape: (height, width, number of classes)
        axis (tuple): axes (height, width) to sum over when computing numerator and
                      denominator for dice loss
        epsilon (float): tiny constant added to numerator and denominator to
                        avoid divide by 0 errors

    Returns:
        dice_subclass3 (float): computed value of dice coefficient
                                for OAAF class  
    '''
    
    dice_numerator = 2 * (K.sum(y_true*y_pred, axis=axis)) + epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
    dice_subclass3 = (dice_numerator/dice_denominator)[:,2]
    
    return dice_subclass3


def preds_to_images(predictions, threshold=0.5):
    
    '''
    Takes raw predictions and converts to RGB format with a probability threshold

    Args:
        predictions (array): predictions in shape (number of images, height, width)
        threshold (float): probability threshold above which to predict a pixel  

    Returns:
        predicted_images (array): predicted images in shape (number of images, height, width)
    '''

    for image_index, image in enumerate(predictions):
        for height_index, height in enumerate(image):
            for width_index, width in enumerate(height):
                max_value = np.max(width)
                predictions[image_index][height_index][width_index] = np.where(width==max_value, max_value, 0)

    predictions[predictions > threshold] = 1.0
    predictions[predictions < threshold] = 0

    predicted_images = 255*predictions

    return predicted_images