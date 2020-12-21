import csv
from bs4 import BeautifulSoup
import urllib
import urllib.request
import re
from urllib.request import urlopen
import os
import tensorflow as tf
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2, NASNetLarge
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

def model():
    #mirrored_strategy = tf.distribute.MirroredStrategy() ###used when using multi-GPU setup. Currently commented since last used on single GPU setup
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) #initialises single GPU for use in training
    total_train = 0
    total_val = 0
    for i in os.listdir('train2'):
        total_train += len(os.listdir(f'train2/{i}')) #count number of training images
    for i in os.listdir('validation2'):
        total_val += len(os.listdir(f'validation2/{i}')) #count number of validation images
    print("Total training images:" + str(total_train))
    print("Total validation images:" + str(total_val))
    train_dir = 'train2' #directory of training images
    validation_dir = 'validation2' #directory of validation images
    batch_size = 8
    epochs = 25
    IMG_HEIGHT = 331
    IMG_WIDTH = 331
    train_image_generator = ImageDataGenerator( #applies image augmentation as well as normalisation of RGB values
                    rescale=1./255, #Image augmentation used is explained further in the report under Machine Learning Development section
                    rotation_range=15,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.15
                    )

    validation_image_generator = ImageDataGenerator(
                    rescale=1./255 #no image augmentation used for validation images since they are used as an indicator of real-world accuracy, very unnecessary
                    )

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse',
                                                           color_mode='rgb')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='sparse',
                                                              color_mode='rgb')
    counter = Counter(train_data_gen.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} #Calculates class weights. Usage is explained further in report
    #with mirrored_strategy.scope(): ###used when using multi-GPU setup. Currently commented since last used on single GPU setup
    model = NASNetLarge(weights = 'imagenet') #use NASNetLarge model with imagenet weights for transfer learning
    def top3_acc(y_true, y_pred): #defines new fucntion that the model returns: returns top 3 accuracy
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
    #model.trainable = True
    #set_trainable = False
    #for layer in model.layers: ###freeze layers before activation_166, layers after activation_166 unfrozen to facilitate faster training and better accuracy of transfer learning
        #if layer.name == 'activation_166': ###currently commented out because test accuracy utilsing this technique was ~0.5% worse
            #set_trainable = True
        #if set_trainable:
            #layer.trainable = True
        #else:
            #layer.trainable = False
    #print("layer {} is {}".format(layer.name, '+++trainable' if layer.trainable else '---frozen'))
    model.compile(optimizer=SGD(0.0005, 0.88, False), loss='sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy',top3_acc]) #compiles model
    model.summary()
    checkpoint_path1 = "checkpoint/NASNetLarge-typesgd8-30-10.h5" #location of model to save to, includes filename. Change filename portion to rename


    checkpoint_dir1 = os.path.dirname(checkpoint_path1)
#    model = load_model(checkpoint_path1)
#    model = load_weights(checkpoint_path1)


    def scheduler(epoch): #learning rate scheduler, decreases learning rate after a certain number of epochs
      if epoch < 7:
        return 0.00001
      else:
        return 0.00001 * tf.math.exp(0.30 * (7 - epoch))

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path1,  #save model with best validation loss after every epoch
                                                 save_weights_only=False,  #if new epoch returns worse validation loss, old model will be retained
                                                 monitor = 'val_loss',
                                                 save_best_only=True,
                                                 verbose=1)
    early_callback = tf.keras.callbacks.EarlyStopping( #model will prematurely stop training if validation accuracy does not increase in 5 epochs
            monitor='val_sparse_categorical_accuracy', min_delta=0.000001, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
        )
#    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#            monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto',
 #           min_delta=0.03, cooldown=2, min_lr=0.000001
#        )

    history = model.fit_generator( #train model
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks = [lr_schedule, cp_callback, early_callback],
        verbose = 1,
	class_weight=class_weights
    )
model()
