"""
Web based Sign Language Predictor 
USAGE:
python webstreaming.py --ip 127.0.0.1 --port 8000

"""

# import all the necessary packages
from ClassifierNet import Net
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import time
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import pymongo

#Initialize the output frame and a lock used to ensure thread-safe
#exchanges of the output frames (useful for multiple browsers/tabs
#are viewing the stream).
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Create Mongo Client using username and password 
client = pymongo.MongoClient("mongodb+srv://meghana-urs:Project123@projectdb-uzrf2.mongodb.net/?retryWrites=true&w=majority")

@app.route("/")
def index():
    """ return the rendered template """
    return render_template("index.html")


def sign_prediction():
    """
    The function predicts sign for every frame of the video using
    the trained model. A square window which contains the hand 
    is cropped from each frame which is the input to the model.
    The predicted letter is displayed on the output frame.

    """
    # Global references to the video stream, output frame, and lock variables
    global vs, outputFrame, lock

    # Load model architecture, pretrained weights and set to eval mode
    model = Net()
    model.load_state_dict(torch.load('./checkpoint.pth'))
    model.eval()

    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    letter = index_to_letter[0]
    print("Inference session")

    previous_timestamp = datetime.datetime.now()
    
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream 
        frame = vs.read()
        (w, h, c) = frame.shape
        s = round(h/2)
        frame = cv2.flip(frame, 1)
        blank_image = np.zeros((w, h, c), np.uint8)

        # Crop the window, convert the frame to grayscale, resize the image 
        blank_image[50: s+50 ,h-50-s: h-50,  :] =  frame[50: s+50 ,h-50-s: h-50,  :]
        model_input = frame[ 50: s+50 ,h-50-s: h-50, :]
        model_input = cv2.flip(model_input, 1)
        model_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(model_input, (28, 28))
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x = x.reshape(1, 1, 28, 28).astype(np.float32)

        # Convert input to Float Tensor and get predictions
        x = torch.FloatTensor(x)
        y = model(Variable(x))
        pred = torch.argmax(y).cpu().numpy()
        letter = index_to_letter[int(pred)]
        timestamp = datetime.datetime.now()

        # Post the prediction to the database after every 15 seconds
        if datetime.datetime.now() >= previous_timestamp + datetime.timedelta(seconds=15):
            previous_timestamp = datetime.datetime.now()

            data = {
                'Time Stamp' : timestamp,
                'Prediction' : letter
            }

            with client:
                db = client.Sign_Prediction
                db.prediction.insert_one(data)

        #Display predictions on the output frame
        frame = cv2.addWeighted(frame, 0.3, blank_image, 1, 0)
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=2)       
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, (h-50-s,50), (h-50,s+50), color, thickness)
        
        with lock:
            outputFrame = frame.copy()


def generate():
    """ Video streaming generator function """
    
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
    """ return the response generated along with the specific media 
    type (mime type)
    """
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

""" check to see if this is the main thread of execution """
if __name__ == '__main__':
    
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=False,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    
    # start a thread that will perform sign prediction

    t = threading.Thread(target=sign_prediction)
    t.daemon = True
    t.start()


    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()