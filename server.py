from copyreg import pickle
from fileinput import filename
import os
import cv2
from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from flask import *
import joblib
from werkzeug.utils import secure_filename
from soil import soilRoutes
import pickle

app = Flask(__name__)
weeds_detection = keras.models.load_model("models/Model.h5")
app.register_blueprint(soilRoutes)

img_height = 180
img_width = 180
class_names_weeds=["broadleaf","grass"]

@app.route('/detectWeed' ,methods=['POST','GET'])
def detect():
    if request.method == "POST":
      file = request.files['image']
      file.save(secure_filename(file.filename))
      filename = secure_filename(file.filename)
      print(filename)
    img = tf.keras.utils.load_img(
        filename, target_size=(img_height, img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch 
    predictions = weeds_detection.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names_weeds[np.argmax(score)], 100 * np.max(score))
    ) 
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                      img_width,
                                      3)),
          layers.RandomRotation(0.1),
          layers.RandomZoom(0.1),
    ]
    )

    return {'status':"{}"
        .format(class_names_weeds[np.argmax(score)], 100 * np.max(score))}
    




if __name__ == "__main__":
  app.run(host="0.0.0.0" ,debug=True, port=5000)

