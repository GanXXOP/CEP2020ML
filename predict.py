import os
import tensorflow as tf
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2, EfficientNetB7, NASNetLarge
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


def evaluate():
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) #use GPU for prediction
    test_dir = 'deletethis' #directory containing test images
    batch_size = 32
    epochs = 20
    IMG_HEIGHT = 331
    IMG_WIDTH = 331

    test_image_generator = ImageDataGenerator( #scale RGB values to between 0-1
                    rescale=1./255
                    )

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='sparse',
                                                              color_mode='rgb')

    checkpoint_path1 = "checkpoint/NASNetLarge-timelinesgd8-30-3.h5" #model filepath

    checkpoint_dir1 = os.path.dirname(checkpoint_path1)
    def top3acc1(y_true, y_pred):  #custom function that returns top 3 accuracy
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
    model = load_model(checkpoint_path1, custom_objects={"top3_acc": top3acc1}) #load model
    model.summary() #optional, prints summary of model and allows user to see the layers in the model architecture
    print(model.metrics_names)
    test_loss, test_acc, top3_acc = model.evaluate(x=test_data_gen, use_multiprocessing=False, verbose=0, max_queue_size=20) #evaluate model accuracy using test set
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)
    print('\nTop 3 accuracy:', top3_acc)


evaluate()

def predict():
    #tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force use CPU for inference instead of GPU, for testing speed of CPU inference
    image_path = 'uploads' #directory for uploaded images to be temporarily stored for inference
    batch_size = 1 #batch size of 1 since there is only 1 photo for inference
    epochs = 20
    IMG_HEIGHT = 331
    IMG_WIDTH = 331

    test_image_generator = ImageDataGenerator(
                    rescale=1./255
                    )

    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=image_path,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode=None,
                                                              color_mode='rgb')

    checkpoint_path1 = "checkpoint/NASNetLarge-timelinesgd8-30-3.h5" #path of model used for inference


    checkpoint_dir1 = os.path.dirname(checkpoint_path1)
    def top3_acc1(y_true, y_pred): #function for top 3 accuracy, required for model to predict
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
    model = load_model(checkpoint_path1, custom_objects={"top3_acc": top3_acc1}) #load model for prediction
    model.summary()
    predictions = model.predict(test_data_gen) #array of predictions
    topclass = np.argmax(predictions[0]) #returns the index of the best performing class
    total = 0
    for i in range(7): #this index is to be changed according to how many classes there are in the category.
                    # If there are 10 classes, the code in the previousshould be ###for i in range(10):###
                    #this is because for transfer learning, we have to use the weights that include the last Dense layer for 1000 classes for predicting the 1000 classes in ImageNet.
                    #Hence we are unable to use our own Dense layer that corresponds to the number of classes.
                    #This function takes the first x elements in the array returned. These correspond to each of the x classes in our category.
                    #The other ~990+ elements return an extremely small prediction on the order of 10^-6 to 10^-7 and is thus negligible, and does not impact our loss or accuracy in any way.
        total += predictions[0][i] #adds the value of the elements together that correspond to a valid class. Ignores the other ~990+ irrelevant elements
    predictions = predictions[0][0:7]/total #Scales the value of the predictions appropriately for elemnents that correspond to a valid class.
                                    #The range of elements to be operated on is to be changed according to how many classes there are in the category.
                                    #If there are 10 classes, the function should be predictions = predictions[0][0:10]/total

    #print('\nTest accuracy:', test_acc)
    print('\nPredictions:', predictions)
    #print('\nTop 3 accuracy:', top3_acc)

#predict()
