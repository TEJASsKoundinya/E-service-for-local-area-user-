
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'InseptionV3error.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds == 2:
        preds = "Company identified APPLE link https://apple-service-center-jp-nagar.business.site/?utm_source=gmb&utm_medium=referral"
    elif preds== 3:
        preds = "Component id as FAN"
    elif preds== 4:
        preds = "Company identified HP link  https://support.hp.com/in-en/service-center "
    elif preds == 5:
        preds = "Component id as LAPTOP B S Computers and Services - 9886918472, 080 - 43940320, 098869 18472"
    elif preds == 9:
        preds = "Company identified LG https://www.lg.com/in/support/telephone?utm_source=SA&utm_medium=cpc&utm_campaign=IN_COR_OTH_IN_21_AO_PHD_PSE_CON_BRA_Service-Center-Exact_lg.com&gclid=Cj0KCQiAwqCOBhCdARIsAEPyW9kusHLKl20aJJHMRNn-NmzGZ9rtWqebO4R0kYIxUzZsQf3bahe_v5waAkv1EALw_wcB"
    elif preds == 6:
        preds = "Component id as LED/ TUBELIGHT reference- Vishal Led Light Service Center- 086180 89505"
    elif preds == 1:
        preds = "Component id as PRINTER reference- https://www.epson.co.in/"
    elif preds == 7:
        preds = "Component id as REMOTE reference- Ramdev- 096110 00362 "
    elif preds == 8:
        preds = "Component id as TV  reference- TV Repair Service Bangalore - 099864 99573 <\n>MAHADESHWARA ELECTRICAL SERVICES AND TV REPAIR - 099640 47052"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('app.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)