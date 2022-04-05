########################################################################
##
##  Use run_train.py to run this script
##  - Defines the convolutional neural network model
##  - Loads images from an hdf5 file
##  - Trains the model on the images with five-fold cross validation
##
########################################################################


import configparser
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lib.helper_functions import load_hdf5
from lib.helper_functions import soft_dice_loss, dice_coefficient, dice_subclass1, dice_subclass2, dice_subclass3
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint


## ResNet-UNet Model with ResNet50 portion pre-trained on ImageNet
def resnet_unet(height, width, n_ch):
    
    inputs = Input(shape=(height, width, n_ch))
    
    resnet = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(height, width, n_ch),
                      input_tensor=inputs)

    for layer in resnet.layers:
        if layer.name.endswith('bn'):
            layer.trainable = False
        else:
            layer.trainable = False

    conv1 = resnet.get_layer("conv1_relu").output
    conv2 = resnet.get_layer("conv2_block3_out").output
    conv3 = resnet.get_layer("conv3_block4_out").output
    conv4 = resnet.get_layer("conv4_block6_out").output
    conv5 = resnet.get_layer("conv5_block3_out").output

    up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv5)
    up1 = concatenate([conv4, up1], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(up1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv6)
    conv6 = Dropout(0.2)(conv6)
    
    up2 = UpSampling2D(size=(2, 2), data_format = 'channels_last')(conv6)
    up2 = concatenate([conv3, up2], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(up2)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv7)
    conv7 = Dropout(0.2)(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv7)
    up3 = concatenate([conv2, up3], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(up3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv8)
    conv8 = Dropout(0.2)(conv8)

    up4 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv8)
    up4 = concatenate([conv1, up4], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(up4)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv9)
    conv9 = Dropout(0.2)(conv9)

    up5 = UpSampling2D(size=(2, 2), data_format='channels_last')(conv9)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_last')(up5)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_last')(conv10)    
    conv10 = Dropout(0.2)(conv10)
    
    conv11 = Conv2D(3, (1, 1), activation='sigmoid', padding='same', data_format='channels_last')(conv10)
    
    model = Model(inputs=inputs, outputs=conv11)
    
    return model


## Load parameters from config file
config = configparser.RawConfigParser()
config.read_file(open(r'./config.txt'))
experiment_name = config.get('experiment name', 'exp_name')
data_file_name = config.get('data file name', 'data_file_name')
result_dir = config.get('result folder name', 'result_dir')

num_epochs = int(config.get('training settings', 'num_epochs'))
learning_rate_initial = float(config.get('training settings', 'learning_rate_initial'))
learning_rate_tune = float(config.get('training settings', 'learning_rate_tune'))
batch_size = int(config.get('training settings', 'batch_size'))
random_state = int(config.get('training settings', 'random_state'))
cross_validation = config.get('training settings', 'cross_validation')

## Load the data, check shapes and fit to ResNet input layer
images, labels = load_hdf5(data_file_name)
print(f'Dimensions of images loaded from hdf5: {images.shape}')
print(f'Dimensions of labels loaded from hdf5: {labels.shape}')
images = np.repeat(images, 3, axis=3)
labels = np.delete(labels, 0, axis=3)

## Train the model
if cross_validation == 'Yes':

    ## Create per-fold metrics lists
    loss_per_fold = []
    dice_per_fold = []
    dice_subclass1_per_fold = []
    dice_subclass2_per_fold = []
    dice_subclass3_per_fold = []
    mean_iou_per_fold = []
    acc_per_fold = []
    
    ## Split into five folds for cross-validation
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 1
    
    ## Run the model for each fold
    for train_index, test_index in kfold.split(images, labels):

        ## Split into training and test sets
        train_images = images[train_index]
        train_labels = labels[train_index]
        
        test_images = images[test_index]
        test_labels = labels[test_index]

        ## Create training generator with image augmentation
        train_image_data_gen_args = dict(featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=30,
                                    width_shift_range=0.25,
                                    height_shift_range=0.25,
                                    shear_range=15,
                                    zoom_range=0.25,
                                    horizontal_flip=True,
                                    data_format='channels_last',
                                    dtype='float32')

        train_label_data_gen_args = dict(rotation_range=30,
                                    width_shift_range=0.25,
                                    height_shift_range=0.25,
                                    shear_range=15,
                                    zoom_range=0.25,
                                    horizontal_flip=True,
                                    data_format='channels_last',
                                    dtype='float32')

        seed = 42

        train_image_data_gen = ImageDataGenerator(**train_image_data_gen_args)
        train_label_data_gen = ImageDataGenerator(**train_label_data_gen_args)

        train_image_data_gen.fit(train_images, augment=True, seed=seed)
        
        train_image_generator = train_image_data_gen.flow(train_images,
                                                          batch_size=batch_size,
                                                          seed=seed)

        train_label_generator = train_label_data_gen.flow(train_labels,
                                                          batch_size=batch_size,
                                                          seed=seed)
        
        train_generator = zip(train_image_generator, train_label_generator)

        ## Standardize test set using mean and standard deviation obtained from the training set
        test_images -= train_image_data_gen.mean
        test_images = test_images / (train_image_data_gen.std + tf.keras.backend.epsilon())

        ## Build the model and save the architecture
        height = train_images.shape[1]
        width = train_images.shape[2]
        n_ch = train_images.shape[3]
        model = resnet_unet(height, width, n_ch)
        json_string = model.to_json()
        open('./' + result_dir + '/' + experiment_name + '_architecture.json', 'w').write(json_string)

        ## Create checkpoint callback
        checkpointer = ModelCheckpoint(filepath='./' + result_dir + '/' + experiment_name + '_fold' + str(fold_num) + '_best_weights.h5', verbose=1, monitor='loss', mode='auto', save_best_only=True) 

        ## Compile the model and do initial fitting
        model.compile(optimizer=RMSprop(learning_rate=learning_rate_initial, centered=True),
                    loss=soft_dice_loss,
                    metrics=[dice_coefficient, dice_subclass1, dice_subclass2, dice_subclass3, tf.keras.metrics.MeanIoU(num_classes=3), 'accuracy'])

        history_initial = model.fit(train_generator,
                            epochs=num_epochs,
                            verbose=2,
                            callbacks=[checkpointer],
                            steps_per_epoch=(len(train_images)//batch_size))

        ## Unfreeze the base layers
        for layer in model.layers:
            if layer.name.endswith('bn'):
                layer.trainable = True
            else:
                layer.trainable = True

        ## Re-compile the model and fine tune with lower learning rate
        model.compile(optimizer=RMSprop(learning_rate=learning_rate_tune, centered=True),
                    loss=soft_dice_loss,
                    metrics=[dice_coefficient, dice_subclass1, dice_subclass2, dice_subclass3, tf.keras.metrics.MeanIoU(num_classes=3), 'accuracy'])

        history_tune = model.fit(train_generator,
                            epochs=num_epochs,
                            verbose=2,
                            callbacks=[checkpointer],
                            steps_per_epoch=(len(train_images)//batch_size))

        ## Save model weights and metrics
        model.save_weights('./' + result_dir + '/' + experiment_name + '_fold' + str(fold_num) + '_best_weights.h5')

        test_metrics = model.evaluate(test_images, test_labels, verbose=1)

        loss_per_fold.append(test_metrics[0])
        dice_per_fold.append(test_metrics[1])
        dice_subclass1_per_fold.append(test_metrics[2])
        dice_subclass2_per_fold.append(test_metrics[3])
        dice_subclass3_per_fold.append(test_metrics[4])
        mean_iou_per_fold.append(test_metrics[5])
        acc_per_fold.append(test_metrics[6])
        
        ## Plot model loss 
        plt.plot(history_initial.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('initial_loss_plot_fold_' + str(fold_num) + '.png')
        plt.clf()

        del history_initial

        plt.plot(history_tune.history['loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('tune_loss_plot_fold_' + str(fold_num) + '.png')
        plt.clf()

        ## Clear the session for the next fold
        del history_tune
        del train_image_generator
        del train_label_generator
        del train_generator
        del model
        del checkpointer
        
        tf.keras.backend.clear_session()

        fold_num += 1

    print(f'Test loss: {loss_per_fold}')
    print(f'Dice coefficients: {dice_per_fold}')
    print(f'Dice subclass 1: {dice_subclass1_per_fold}')
    print(f'Dice subclass 2: {dice_subclass2_per_fold}')
    print(f'Dice subclass 3: {dice_subclass3_per_fold}')
    print(f'Mean IOU: {mean_iou_per_fold}')
    print(f'Accuracy: {acc_per_fold}')
    