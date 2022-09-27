from flask import *
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
from werkzeug.utils import secure_filename




soilRoutes=Blueprint("routes ",__name__)
soil_detection = keras.models.load_model("models/soil_Model.h5")
class_names_soil=["Black soil","cinder soil","Laterite Soil","peat Soil","yellow Soil"]
img_height = 180
img_width = 180
@soilRoutes.route('/detectSoil' ,methods=['POST','GET'])
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
    predictions = soil_detection.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names_soil[np.argmax(score)], 100 * np.max(score))
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
    
    return  { 'status':"{}"
        .format(class_names_soil[np.argmax(score)], 100 * np.max(score))}


