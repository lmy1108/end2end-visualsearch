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
from urllib import unquote,quote
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
app = flask.Flask(__name__)
CORS(app, supports_credentials=True)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

BATCH_SIZE = 256
EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])



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

def prepare_image2(image, target):

	# if the image mode is not RGB, convert it
	# if image.mode != "RGB":
	# 	image = image.convert("RGB")
	image.save('./plt0.png')
	# resize the input image and preprocess it
	image_array = img_to_array(image)
	print("type of image:" , type(image_array))
	x, img = gcv.data.transforms.presets.ssd.load_test('plt0.png',short=1024)

	fig = plt.gcf()
	# fig.set_size_inches(18.5, 10.5)
	ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
	ssdnet = get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
	classes, scores, bbox = ssdnet(x.as_in_context(ctx))
	viz.plot_bbox(img, bbox[0], scores[0], classes[0], class_names=ssdnet.classes)
	fig = plt.gcf()
	# fig.set_size_inches(18.5, 10.5)
	plt.show()
	plt.savefig('static/images/plt1.jpg', dpi=100)


	bbout = bbox[0][0].asnumpy()
	x0 = int(bbout[0])
	x1 = int(bbout[2])
	y0 = int(bbout[1])
	y1 = int(bbout[3])
	cropped_img = img[y0:y1, x0:x1, :]
	viz.plot_image(cropped_img)
	fig = plt.gcf()
	fig.set_size_inches(6.5, 6.5)
	plt.show()
	plt.savefig('static/images/plt2.jpg', dpi=100)

	pspnet = get_model('FCN_resnet50_ade', pretrained=True, ctx=ctx)

	from PIL import Image
	im = Image.fromarray(cropped_img)
	x, cropped_loaded = gcv.data.transforms.presets.ssd.transform_test(nd.array(cropped_img),
																  min(cropped_img.shape[1], cropped_img.shape[0]))
	output = pspnet.demo(x.as_in_context(ctx))
	pred = output.argmax(1).asnumpy().squeeze()

	mask = viz.get_color_pallete(pred, 'ade20k')
	mask = np.array(mask.convert('RGB'), dtype=np.int)

	combined = (mask + cropped_img) / 2

	plt.imshow(combined.astype(np.uint8))
	fig = plt.gcf()
	fig.set_size_inches(6.5, 6.5)
	plt.show()
	plt.savefig('static/images/plt3.jpg', dpi=100)
	HUMAN = 12
	filtered = (pred == HUMAN)
	filtered = filtered.reshape(filtered.shape[0],filtered.shape[1], 1).repeat(3, 2)
	filtered_img = filtered * cropped_img
	filtered_img[filtered == False] = 255
	viz.plot_image(filtered_img)
	fig = plt.gcf()
	fig.set_size_inches(6.5, 6.5)
	plt.show()
	plt.savefig('static/images/plt4.jpg', dpi=100)
	image = Image.open('VisualSearch_MXNet_CFN-master/web/plt4.jpg')
	image = image.resize(target)
	image = img_to_array(image)

	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	plt.cla()
	plt.close("all")
	return image
@app.route('/index/')
def hello():
	return render_template('index.html')

@app.route("/")
def homepage():
	return "Welcome to the PyImageSearch Keras REST API!"


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint

	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		data_decode = unquote(data1)
		json_re = json.loads(data_decode)
		print("json_re is" , json_re)

		print(type(json_re))
		imgdata = base64.b64decode(data1)
		print(type(imgdata))

		if type(imgdata) is str:
			# read the image in PIL format and prepare it for
			# classification



			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')

			print(type(image))

			image = prepare_image(image,
				(500, 500))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					# data["predictions"] = json.loads(output)
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


@app.route("/predict2", methods=["POST"])
def predict2():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint

	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		data_decode = unquote(data1)
		json_re = json.loads(data_decode)
		print("json_re is" , json_re)

		print(type(json_re))
		imgdata = base64.b64decode(data1)
		print(type(imgdata))

		if type(imgdata) is str:
			# read the image in PIL format and prepare it for
			# classification



			image = Image.open(io.BytesIO(imgdata))
			print(type(image))
			image.save('raw.png')

			image = prepare_image2(image,
				(500, 500))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					# data["predictions"] = json.loads(output)
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)
# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	print("* Starting web service...")
	app.run()