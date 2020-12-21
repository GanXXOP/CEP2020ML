import csv
from bs4 import BeautifulSoup
import urllib
import urllib.request
import re
from urllib.request import urlopen
import os
import wget
import numpy as np
import matplotlib.pyplot as plt
import keras
from random import randint
from shutil import copyfile
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def filter1():
    filtered = []
    with open('catalog.csv', newline='') as csvfile:
        alldata = csv.reader(csvfile, delimiter=",")
        for line in alldata:
            if line[7] == "painting":
                filtered.append(line)
    return filtered
#this function extracts all paintings and outputs a list. any other types of artwork ignored

def scrape():
    data = filter1()
    imageurl = []
    for i in data:
        webpage = "https://www.wga.hu/detail"+str(i[6][23:-4])+("jpg")
        imageurl.append(webpage)
    return(imageurl)
#this fuction retuns a list of webpages where images of paintings are located

def download():
    allurl = scrape()
    data = filter1()
    if not os.path.exists('images'):
        os.makedirs('images')
    for i in range(len(allurl)):
        if os.path.isfile(f"images/{i}.jpg"):
            pass #check if image is already downloaded, skips if downloaded
        else:
            name = str(filter1()[i][2])+".jpg"
            urllib.request.urlretrieve(allurl[i], f"images/{i}.jpg")
            print(i)
            print(allurl[i])
#this fuction iterates through list returned by scrape() and downloads images that are still not downloaded yet


def sort():
    download()
    list1 = os.listdir("images")
    number_files = len(list1)
    data = filter1()
    if not os.path.exists('train5'): #makes new training, validation and test directories if they do not exist
        os.makedirs('train5')
    if not os.path.exists('validation5'):
        os.makedirs('validation5')
    if not os.path.exists('test5'):
        os.makedirs('test5')
    for i in range(len(list1)):
        number = randint(1,20000) #random number generator
        if (number % 20)<3 and int(data[i][10][:4])>1400: #integer from 0-2 after modulo 20 to validation folder, year must be bigger than 1400
            year = (data[i][9]) #index 8 for art genre/type, 9 for art school, 10 for timeline as per column number in CSV file
            if not os.path.exists(f'validation5/{year}'): #make new class folder if missing, copies image to the new class
                os.makedirs(f'validation5/{year}')
                copyfile(f"images/{i}.jpg", f"validation5/{year}/{i}.jpg")
            else:
                if not os.path.isfile(f"validation5/{year}/{i}.jpg"): #copies image to the new class
                    copyfile(f"images/{i}.jpg", f"validation5/{year}/{i}.jpg") #copies image to the new class
                pass
        elif 13<(number % 20)<17 and int(data[i][10][:4])>1400: #integer from 14-16 after modulo 20 to test folder, year must be bigger than 1400
            year = (data[i][9]) #index 8 for art genre/type, 9 for art school, 10 for timeline as per column number in CSV file
            if not os.path.exists(f'test5/{year}'): #make new class folder if missing, copies image to the new class
                os.makedirs(f'test5/{year}')
                copyfile(f"images/{i}.jpg", f"test5/{year}/{i}.jpg")
            else:
                if not os.path.isfile(f"test5/{year}/{i}.jpg"): #copies image to the new class
                    copyfile(f"images/{i}.jpg", f"test5/{year}/{i}.jpg")
                pass
        else:
            year = (data[i][9]) #index 8 for art genre/type, 9 for art school, 10 for timeline as per column number in CSV file
            if not os.path.exists(f'train5/{year}') and int(data[i][10][:4])>1400: #make new class folder if missing, copies image to the new class
                os.makedirs(f'train5/{year}')
                copyfile(f"images/{i}.jpg", f"train5/{year}/{i}.jpg")
            elif int(data[i][10][:4])>1400: #year must be bigger than 1400
                if not os.path.isfile(f"train5/{year}/{i}.jpg"): #copies image to the new class
                    copyfile(f"images/{i}.jpg", f"train5/{year}/{i}.jpg")
                pass

sort()

#for plotting of graph, irrelevant to final model
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


#initial image classification architecture design, outdated
#view nasnet.py for the final training code and more extensive comments
def model():
    total_train = 0
    total_val = 0
    for i in os.listdir('train'):
        total_train += len(os.listdir(f'train/{i}'))
    for i in os.listdir('validation'):
        total_val += len(os.listdir(f'validation/{i}'))
    print("Total training images:" + str(total_train))
    print("Total validation images:" + str(total_val))
    train_dir = 'train'
    validation_dir = 'validation'
    batch_size = 256
    epochs = 20
    IMG_HEIGHT = 400
    IMG_WIDTH = 400
    train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.15
                    )

    validation_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           color_mode='rgb')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode='rgb',
                                                              class_mode='categorical')
##    sample_training_images, _ = next(train_data_gen)
##    plotImages(sample_training_images[:5])

    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    model = Sequential([
        Conv2D(8, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    #how these layers work is explained in the Written Report under Machine Learning Algorithm section
    model.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics = ['accuracy','categorical_accuracy'])
    model.summary()
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



#sort()
