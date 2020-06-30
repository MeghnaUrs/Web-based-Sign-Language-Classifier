# Web-based-Sign-Language-Classifier
***Introduction***

&nbsp;&nbsp;&nbsp;Sign Languages facilitate communication using hand signs. The earliest usage of sign languages has been recorded to be 5000 BC. In the research published by Centers for Disease Control and Prevention (CDC) in 2010, around 3 out of every 1000 children in the United States of America are born with a detectable amount of hearing loss in one or both ears. This project is a humble effort to help practice Sign Language Alphabets using Deep Learning models.

***Problem Statement***

&nbsp;&nbsp;&nbsp;Deploy a deep learning-based web app capable of predicting and displaying the sign language alphabet from the webcam video feed of the Client. 

***Approach***

&nbsp;&nbsp;&nbsp;As a part of data collection, I used the Kaggle dataset (present here) consisting of 27,455 training and 7172 test images. After training the model for 20 epochs, the accuracy of the model on the training set is 99.5% and on the test set is 96%.

A custom model as shown below has been used for this task:
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/43301609/86103828-1a63af00-ba72-11ea-9ce8-a50856353bc3.png">
</p>

&nbsp;&nbsp;&nbsp;Once trained the Pytorch model is integrated with the Flask based web App. Predictions and their time stamp are stored every 15 seconds on cloud-based Mongo DB server. The front-end of the web app has been developed using bootstrap theme “Simply Me” borrowed from W3 schools(Link). Images used in this project were downloaded from Google Search Engine.

Note: Due to the limitation in the diversity of the training data, the model works best with a clean background.

***Setup and Installation***

Step 0: ```Git clone this project```<br>
Step 1: ```cd <local-repository-folder>``` <br>
Step 2: ```pip install Requirements.txt```<br>
Step 3: ```python web-streaming.py –ip 0.0.0.0 –host 8000 (For MacOS/ Linux )```<br>
        ```python web-streaming.py –ip 127.0.0.1 –host 8000 (For Windows )```

Tools: Pytorch, OpenCV, Flask, Mongo DB (PyMongo), Bootstrap 

***Future Enhancements***<br><br>
&nbsp;&nbsp;&nbsp;Heroku based online website in progress…

