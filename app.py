from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import tensorflow as tf
#print(tf.version.VERSION)
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#model_path = 'brain_model.hdf5'

# Load your trained model
model = load_model('brain_model_final.h5')

def get_label(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
            return key



def make_pred(img_path, model):
    categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    label=[i for i in range(len(categories))]
    label_dict=dict(zip(categories,label))
    img = image.load_img(img_path, target_size = (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.array(img_array)/255
    img_array = np.expand_dims(img_array, axis = 0)
    y_pred = model.predict(img_array)
    y = np.argmax(y_pred)
    label = get_label(y, label_dict)
    return label


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('upload.html')


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
        preds = make_pred(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)