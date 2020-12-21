# Y4 Final Project Front End #

Prerequisites to be installed:

tf-nightly==2.3.0.dev20200624
pillow
sqlalchemy
flask
flask_session

These can be installed via pip commands.

Information such as image classes used and the accuracy of the model can be found in the Report PDF.

The entire project including images used and the file structure required for the website to work can be found at https://drive.google.com/file/d/1cd90Zy4GGK05izpyU8MrDPaU7isrbUzu/view

Reference images and categories in our project can be found at the folder "images" for all the downloaded, unsorted images.
Training, validation and test sets used in our proejct are found at the various folders labelled with training, test and validation in their filenames. 
The final training models to be used for inference can be found in the checkpoint folder.

This is the front-end documentation of our art identification program.

To run application.py (the application), you will need to set 2-3 environment variables after navigating to the directory that application.py is in.
These are the relevant commands:

1a (If using Powershell) $env:FLASK_APP = "application.py"
1b (If using Command Prompt) set FLASK_APP = application.py
1c (If using Mac) export FLASK_APP = application.py

(Optional)
2a (If using Powershell) $env:FLASK_DEBUG = 1
2b (If using Command Prompt) set FLASK_DEBUG = 1
2c (If using Mac) export FLASK_DEBUG = 1

(IMPORTANT - This is the database with all user data in it)
3a (If using Powershell) $env:DATABASE_URL = "postgres://trywaaukfhhecs:abfbb928be97641b94a827df3496ca06c24a1b518d0cca3b5237bdebc5b71131@ec2-52-6-143-153.compute-1.amazonaws.com:5432/d9e2oha1sh2bjv"
3b (If using Command Prompt) set DATABASE_URL = postgres://trywaaukfhhecs:abfbb928be97641b94a827df3496ca06c24a1b518d0cca3b5237bdebc5b71131@ec2-52-6-143-153.compute-1.amazonaws.com:5432/d9e2oha1sh2bjv
3c (If using Mac) export DATABASE_URL = "postgres://trywaaukfhhecs:abfbb928be97641b94a827df3496ca06c24a1b518d0cca3b5237bdebc5b71131@ec2-52-6-143-153.compute-1.amazonaws.com:5432/d9e2oha1sh2bjv

Finally, to begin running the application, just input "flask run" (excluding the inverted commas) and copy the link provided in the console
onto a browser's URL bar. This will usually be 127.0.0.1:5000. 
