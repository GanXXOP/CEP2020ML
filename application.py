DATABASE_URL = "postgres://trywaaukfhhecs:abfbb928be97641b94a827df3496ca06c24a1b518d0cca3b5237bdebc5b71131@ec2-52-6-143-153.compute-1.amazonaws.com:5432/d9e2oha1sh2bjv"

import os
import json
import requests
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import glob
from info import * #py file containing informative text

from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from flask import Flask, session, render_template, url_for, redirect, request, escape, flash
from flask_session import Session
from flask import send_from_directory
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from werkzeug.utils import secure_filename


if not os.path.exists('uploads'):
    os.makedirs('uploads')
UPLOAD_FOLDER = "uploads"
#Path to folder storing user uploads
if not os.path.exists('uploads/temp'):
    os.makedirs('uploads/temp')
TEMP = "uploads/temp"
#Temporary folder within upload folder, for machine learning purposes

#tensorflow potion
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#model paths
checkpoint_path1 = "checkpoint/NASNetLarge-countrysgd8-20-3.h5"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)

checkpoint_path2 = "checkpoint/NASNetLarge-timelinesgd8-30-3.h5"
checkpoint_dir2 = os.path.dirname(checkpoint_path2)

checkpoint_path3 = "checkpoint/NASNetLarge-typesgd8-30-3.h5"
checkpoint_dir3 = os.path.dirname(checkpoint_path3)

def top3_acc1(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)

#preload tensorflow models
model1 = load_model(checkpoint_path1, custom_objects={"top3_acc": top3_acc1})
pmodel1 = tf.keras.Sequential([model1])
model2 = load_model(checkpoint_path2, custom_objects={"top3_acc": top3_acc1})
pmodel2 = tf.keras.Sequential([model2])
model3 = load_model(checkpoint_path3, custom_objects={"top3_acc": top3_acc1})
pmodel3 = tf.keras.Sequential([model3])


#image prediction
def predict():
    image_path = 'uploads'
    IMG_HEIGHT = 331
    IMG_WIDTH = 331
    batch_size = 1
    test_image_generator = ImageDataGenerator(
                    rescale=1./255
                    )
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                    directory=image_path,
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    class_mode=None,
                    color_mode='rgb')


    predictions1 = pmodel1.predict(test_data_gen)
    topclass1 = np.argmax(predictions1[0])
    total1 = 0
    for i in range(6):
        total1 += predictions1[0][i]
    predictions1 = predictions1[0][0:6]/total1
    #predictions1 is the predictions for the art school/country

    predictions2 = pmodel2.predict(test_data_gen)
    topclass2 = np.argmax(predictions2[0])
    total2 = 0
    for i in range(10):
        total2 += predictions2[0][i]
    predictions2 = predictions2[0][0:10]/total2
    #predictions2 is the predictions for the art era/period

    predictions3 = pmodel3.predict(test_data_gen)
    topclass3 = np.argmax(predictions3[0])
    total3 = 0
    for i in range(7):
        total3 += predictions3[0][i]
    predictions3 = predictions3[0][0:7]/total3
    #predictions2 is the predictions for the art era/period

    return [predictions1, predictions2, predictions3]


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#Allowed filetypes, in this case only images.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#check for file validity

app = Flask(__name__)

# Check for environment variable
#if not os.getenv("DATABASE_URL"):
    #raise RuntimeError("DATABASE_URL is not set")

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP'] = TEMP
Session(app)

# Set up database
engine = create_engine(DATABASE_URL)
db = scoped_session(sessionmaker(bind=engine))

@app.route("/")
def index(): #Welcome page, tells you to login
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register(): #Registration page
    error = None #Error message that may or may not be used
    if request.method == "GET":
        return render_template("register.html")
    else:
        username = escape(request.form.get("username")) #Takes user's input and sanitizes input
        password = escape(request.form.get("password"))

        if username == "": #Prevents blank space or NULL usernames
            return render_template("register.html", error="Input a valid username!")

        elif db.execute("SELECT * FROM users WHERE username = :username", {"username":username}).rowcount == 0: #If username not taken
            db.execute("INSERT INTO users (username, password) VALUES (:username, :password)", {"username": username, "password": password})
            db.commit() #Inputs user's username and password into database
            return redirect(url_for("login")) #redirect to login page

        else: #If username already exists
            return render_template("register.html", error="That username is already taken!")

@app.route("/login", methods=["GET","POST"])
def login(): #Login page
    fail_msg = None
    if request.method == "GET":
        return render_template("login.html")

    else:
        username = escape(request.form.get('username')) #escape() sanitizes input
        password = escape(request.form.get('password'))

        #Check if username exists and corresponding password matches
        if db.execute("SELECT * FROM users WHERE username = :username and password = :password", {"username": username, "password": password}).rowcount == 1:
            session['username'] = username #Stores the user's name inside the session, keeps track of who is logged in
            return redirect(url_for("search"))
        else: #If wrong username/password
            return render_template("login.html", fail_msg = "Invalid username / password! Please try again!")

@app.route("/search", methods=["GET","POST"])
def search(): #Search & Upload page
    fail_msg = ""
    if 'username' not in session: #Prevents users who have not logged in from accessing pages
        return render_template("loginerror.html")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            fail_msg = "No file part!"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            fail_msg = "No selected file!"
        if file and allowed_file(file.filename):
            if not os.path.exists(app.config['TEMP']):
                os.makedirs(app.config['TEMP'])
            shutil.rmtree(os.path.join(app.config['TEMP']))
            if not os.path.exists(app.config['TEMP']):
                os.makedirs(app.config['TEMP'])
            filename = str(secure_filename(file.filename))
            path = os.path.join(app.config['TEMP'])
            file.save(os.path.normpath(os.path.join(path, filename)))
            #Saves user input to upload folder
            returnvalue = predict()
            #runs the uploaded image through the predict function
            #obtains the predictions, and assigns them to the class and sorts according to accuracy
            results1 = {"Dutch":100*returnvalue[0][0], "Flemish":100*returnvalue[0][1], "French":100*returnvalue[0][2],
                    "German":100*returnvalue[0][3], "Italian":100*returnvalue[0][4], "Spanish":100*returnvalue[0][5]}
            #assigns a the prediction value to a class
            results1 = sorted(results1.items(), key=lambda x: x[1], reverse=True) #sorts in desending order of values
            results2 = {"1401-1450":100*returnvalue[1][0], "1451-1500":100*returnvalue[1][1], "1501-1550":100*returnvalue[1][2],
                    "1551-1600":100*returnvalue[1][3], "1601-1650":100*returnvalue[1][4], "1651-1700":100*returnvalue[1][5],
                    "1701-1750":100*returnvalue[1][6], "1751-1800":100*returnvalue[1][7], "1801-1850":100*returnvalue[1][8], "1851-1900":100*returnvalue[1][9]}
            #assigns a the prediction value to a class
            results2 = sorted(results2.items(), key=lambda x: x[1], reverse=True) #sorts in desending order of values
            results3 = {"Genre":100*returnvalue[2][0], "Historical":100*returnvalue[2][1], "Landscape":100*returnvalue[2][2],
                    "Mythological":100*returnvalue[2][3], "Portrait":100*returnvalue[2][4], "Religious":100*returnvalue[2][5], "Still life":100*returnvalue[2][6]}
            results3 = sorted(results3.items(), key=lambda x: x[1], reverse=True) #sorts in desending order of values

            fail_msg = "" #No fail message as input is successful
            return render_template("results.html", filename=filename, results1=results1, results2=results2, results3=results3)

    #If something goes wrong, returns same page with relevant fail message
    return render_template("search.html", error = fail_msg)

@app.route("/info/<school>", methods=["GET"])
def info(school):
    info = ""
    source = ""
    #Loads relevant informative text and source given the page accessed by user
    if school == 'Dutch': #Allows user to learn about Dutch art
        info = Dutch
        source = Dutch_Source
    if school == 'Flemish': #etc for other schools of art
        info = Flemish
        source = Flemish_Source
    if school == 'French':
        info = French
        source = French_Source
    if school =='German':
        info = German
        source = German_Source
    if school == 'Italian':
        info = Italian
        source = Italian_Source
    if school == 'Spanish':
        info = Spanish
        source = Spanish_Source
    if school == 'ML':#Allows user to learn about our machine learning framework
        info = ML

    return render_template("{}.html".format(school), info=info, source=source)

@app.route("/sample/<category>", methods=["GET"])
def sample(category):
    info = ""
    #Loads relevant informative text given the page accessed by user
    if category == 'Portrait':
        info = Portrait
    if category == 'Landscape':
        info = Landscape
    if category == 'Still Life':
        info = Still_Life
    if category == 'Genre':
        info = Genre

    return render_template("{}_sample.html".format(category), info=info)

@app.route("/logout")
def logout(): #Logout
    session.pop('username', None) #Removes the user's name from the session
    return redirect(url_for('index')) #Returns to welcome/start page

@app.errorhandler(404)
def page_not_found(e): #Allows logout or navigation back to main search page when user accesses a non-existent page.
    return render_template('404.html'), 404
