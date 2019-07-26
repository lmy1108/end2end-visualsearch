# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import helpers
import flask
from flask import render_template
import redis
import uuid
import time
import json
import io
from flask_cors import *
import urllib.parse
import base64

import mxnet as mx
import numpy as np
import gluoncv as gcv
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz, download
# initialize our Flask application and Redis server

import chest
import joblib
from keras.applications import resnet50
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
import keras
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

app = flask.Flask(__name__)
CORS(app, supports_credentials=True)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

BATCH_SIZE = 256
EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])
from sklearn.metrics import accuracy_score
import joblib

estimator = RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
# estimator.fit(X_train, y_train)
print("loading")
# X_train = pickle.load(open('X_train', 'rb'))
# X_test = pickle.load(open('X_test', 'rb'))
# y_train = pickle.load(open('y_train', 'rb'))
# y_test = pickle.load(open('y_test', 'rb'))
# from sklearn.metrics import accuracy_score, classification_report
#
# estimator = RandomForestClassifier(n_estimators=500, random_state=123, n_jobs=-1)
# estimator.fit(X_train, y_train.astype('int'))
estimator=joblib.load('estimator_100')

# y_pred = estimator.predict(X_test)
# print(accuracy_score(y_test, y_pred))
#
# print(classification_report(y_test, y_pred))
# joblib.dump(estimator, filename='RandomForestClassifier_new')

print("load succeeded!")
image_size = 224
n_channels = 3
resmodel = resnet50.ResNet50(include_top=False,
                              weights='imagenet',
                              pooling=None,
                              input_shape=(224, 224, 3))



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image
def transform(image, label):
	resized = mx.image.resize_short(image, SIZE[0]).astype('float32')
	cropped, crop_info = mx.image.center_crop(resized, SIZE)
	cropped /= 255.
	normalized = mx.image.color_normalize(cropped,
                                      mean=MEAN_IMAGE,
                                      std=STD_IMAGE)
	transposed = nd.transpose(normalized, (2,0,1))
	return transposed, label


@app.route('/index/')
def hello():
	return render_template('index.html')

@app.route("/")
def homepage():
	return "Welcome to the PyImageSearch Keras REST API!"


@app.route("/classify", methods=["POST"])
def classify():
	# initialize the data dictionary that will be returned from the
	# view
	keras.backend.clear_session()
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint

	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		# data_decode = urllib.parse(data1)
		# json_re = json.loads(data_decode)
		# print("json_re is" , json_re)
		#
		# print(type(json_re))
		imgdata = base64.b64decode(data1)
		print(type(imgdata))

		if type(imgdata) is not None:
			# read the image in PIL format and prepare it for
			# classification



			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')

			print(type(image))
			resmodel = resnet50.ResNet50(include_top=False,
										 weights='imagenet',
										 pooling=None,
										 input_shape=(224, 224, 3))
			# image = Image.open('clothes.jpg')
			image = prepare_image(image,(224, 224))
			tempArray = resmodel.predict(image)

			result_proba = estimator.predict_proba(tempArray[0][0])
			print(result_proba)
			result_label = pickle.load(open('result_label.pkl', 'rb'))

			mapping = {}
			for i in range(0, result_proba[0].size):
				mapping[result_label[i]] = result_proba[0][i]
			print(mapping)
			data["predictions"] = mapping


			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)






if __name__ == "__main__":
	print("* Starting web service...")
	app.run(port=5010)