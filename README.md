# Web-based-Sign-Language-Classifier
***Introduction***

&nbsp;&nbsp;&nbsp;Sign Languages facilitate communication using hand signs. The earliest usage of sign languages has been recorded to be 5000 BC. In the research published by Centers for Disease Control and Prevention (CDC) in 2010, around 3 out of every 1000 children in the United States of America are born with a detectable amount of hearing loss in one or both ears. This project is a humble effort to help practice Sign Language Alphabets using Deep Learning model based architecture.

***Problem Statement***

&nbsp;&nbsp;&nbsp;Deploy a deep learning-based web app capable of predicting and displaying the sign language alphabet from the webcam video feed of the Client. 

***Approach***

&nbsp;&nbsp;&nbsp;As a part of data collection, I used the <a href="https://www.kaggle.com/datamunge/sign-language-mnist">Kaggle dataset<a> consisting of 27,455 training and 7172 test images. After training the model for 20 epochs, the accuracy of the model on the training dataset is 99.5% and on the test dataset is 96.8%.

A custom CNN architecture based model as shown below has been used for this task:
<p align="center">
  <img width="1000" height="300" src="https://user-images.githubusercontent.com/43301609/86103828-1a63af00-ba72-11ea-9ce8-a50856353bc3.png">
</p>

&nbsp;&nbsp;&nbsp;Once trained, the Pytorch model is integrated with the Flask based web App. Predictions and their time stamp are stored every 15 seconds on cloud-based Mongo DB server. The front-end of the web app has been developed using bootstrap theme “Simply Me” borrowed from <a href="https://www.w3schools.com/bootstrap/bootstrap_theme_me.asp">W3 schools<a>. Images used in this project were downloaded from Google Search Engine.

***Note: Due to the limitation in diversity of the training dataset, the model works best with a clean, plain background.***

***Setup and Installation***
Change torch wheel file in requirements.txt based on Windows, Linux or Mac <a href="https://pytorch.org/get-started/locally/#anaconda">Check heret<a>
Step 0: ```Git clone this project```<br>
Step 1: ```cd <local-repository-folder>``` <br>
Step 2: ```pip3 install -r requirements.txt```<br>
Step 3: ```python web-streaming.py –ip 0.0.0.0 –host 8000 (For MacOS/ Linux )```<br>
        ```python web-streaming.py –ip 127.0.0.1 –host 8000 (For Windows )```<br>

Tools: Pytorch, OpenCV, Flask, Mongo DB (PyMongo), Bootstrap 

***Future Enhancements***<br><br>
&nbsp;&nbsp;&nbsp;Heroku based online website in progress…

