# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import pymongo
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
app = flask.Flask(__name__)
CORS(app, supports_credentials=True)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

BATCH_SIZE = 256
EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])

myclient = pymongo.MongoClient("mongodb://192.168.156.152:27017/")
mydb = myclient["product"]
mycol = mydb["datas"]

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
	start1 = time.time()
	# if the image mode is not RGB, convert it
	# if image.mode != "RGB":
	# 	image = image.convert("RGB")
	image.save('./plt0.png')
	# resize the input image and preprocess it
	image_array = img_to_array(image)
	print("type of image:" , type(image_array))
	x, img = gcv.data.transforms.presets.ssd.load_test('plt0.png',short=1024)
	end1 = time.time()
	start2 = time.time()

	fig = plt.gcf()
	# fig.set_size_inches(18.5, 10.5)
	ctx = mx.gpu()
	ssdnet = get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=ctx)
	classes, scores, bbox = ssdnet(x.as_in_context(ctx))
	viz.plot_bbox(img, bbox[0], scores[0], classes[0], class_names=ssdnet.classes)
	fig = plt.gcf()
	# fig.set_size_inches(18.5, 10.5)
	plt.show()
	plt.savefig('static/images/plt1.jpg', dpi=100)
	end2 = time.time()

	start3 = time.time()

	# bbout = bbox[0][0].asnumpy()
	# 	# x0 = int(bbout[0])
	# 	# x1 = int(bbout[2])
	# 	# y0 = int(bbout[1])
	# 	# y1 = int(bbout[3])
	# 	# cropped_img = img[y0:y1, x0:x1, :]
	# 	# viz.plot_image(cropped_img)
	# 	# fig = plt.gcf()
	# 	# fig.set_size_inches(10.5, 18.5)
	# 	# plt.show()
	# 	# plt.savefig('static/images/plt2.jpg', dpi=100)

	end3 = time.time()

	cropped_img = img
	pspnet = get_model('FCN_resnet50_ade', pretrained=True, ctx=ctx)

	from PIL import Image
	im = Image.fromarray(cropped_img)
	x, cropped_loaded = gcv.data.transforms.presets.ssd.transform_test(nd.array(cropped_img),
																  min(cropped_img.shape[1], cropped_img.shape[0]))
	start4 = time.time()

	output = pspnet.demo(x.as_in_context(ctx))
	pred = output.argmax(1).asnumpy().squeeze()
	end4 = time.time()

	mask = viz.get_color_pallete(pred, 'ade20k')
	mask = np.array(mask.convert('RGB'), dtype=np.int)

	combined = (mask + cropped_img) / 2

	plt.imshow(combined.astype(np.uint8))
	fig = plt.gcf()
	fig.set_size_inches(10.5, 18.5)
	plt.show()
	plt.savefig('static/images/plt3.jpg', dpi=100)
	# end4 = time.time()
	start5 = time.time()

	HUMAN = 12
	filtered = (pred == HUMAN)
	filtered = filtered.reshape(filtered.shape[0],filtered.shape[1], 1).repeat(3, 2)
	filtered_img = filtered * cropped_img
	filtered_img[filtered == False] = 255
	viz.plot_image(filtered_img)
	fig = plt.gcf()
	fig.set_size_inches(5.5, 9)
	plt.axis('off')
	plt.show()
	plt.savefig('static/images/plt4.jpg', dpi=200)
	end5 = time.time()

	curimage = Image.open('static/images/plt4.jpg')
	cursize = curimage.size
	x = cursize[0]

	y = cursize[1]
	curimage = curimage.crop((x/4, y/4, 3*x/4, 3*y/4))
	print("图片宽度和高度分别是{}".format(curimage.size))
	# curimage.save('static/images/plt5.png')
	curimage = curimage.resize(target)
	curimage = img_to_array(curimage)

	curimage = np.expand_dims(curimage, axis=0)
	curimage = imagenet_utils.preprocess_input(curimage)
	plt.cla()
	plt.close("all")

	print("1",end1 - start1,"2",end2 - start2,"3",end3 - start3,"4",end4 - start4,"5",end5 - start5)
	return curimage

def predictTaghelper(image):
	image.save('./plt0.png')
	# resize the input image and preprocess it
	image_array = img_to_array(image)
	print("type of image:", type(image_array))
	x, img = gcv.data.transforms.presets.ssd.load_test('plt0.png', short=1024)

	fig = plt.gcf()
	# fig.set_size_inches(18.5, 10.5)
	ctx = mx.gpu()
	ssdnet = get_model('yolo3_darknet53_coco', pretrained=True, ctx=ctx)
	classes, scores, bbox = ssdnet(x.as_in_context(ctx))
	# viz.plot_bbox(img, bbox[0], scores[0], classes[0], class_names=ssdnet.classes)
	classlist = classes.asnumpy().tolist()
	scoreslist = scores.asnumpy().tolist()
	s = set()
	sname = []
	for i in range(0, len(scoreslist[0])):
		if scoreslist[0][i][0] > 0.01 and classlist[0][i][0] not in s:
			s.add(classlist[0][i][0])
			sname.append(ssdnet.classes[int(classlist[0][i][0])]+":" +str(round(scoreslist[0][i][0],2)))
	# for i in s:
	# 	sname.append(ssdnet.classes[int(i)]+"\:" + ssdnet.scores[int(i)])

	print("tags names are:",sname)
	return sname

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
		# data_decode = unquote(data1)
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

			image = prepare_image(image,
				(500, 500))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'general'}
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

@app.route("/predictbooks", methods=["POST"])
def predictbooks():
	data = {"success": False}
	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		imgdata = base64.b64decode(data1)
		print(type(imgdata))
		if type(imgdata) is not None:
			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')
			print(type(image))
			image = prepare_image(image,
				(500, 500))
			image = image.copy(order="C")
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'books'}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			# predictions
			while True:
				output = db.get(k)
				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(k)
					break
				time.sleep(settings.CLIENT_SLEEP)
			data["success"] = True
	return flask.jsonify(data)

@app.route("/predictClothing", methods=["POST"])
def predictClothing():
	data = {"success": False}
	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		imgdata = base64.b64decode(data1)
		print(type(imgdata))
		if type(imgdata) is not None:
			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')
			print(type(image))
			image = prepare_image(image,
				(500, 500))
			image = image.copy(order="C")
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'Clothing'}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			# predictions
			while True:
				output = db.get(k)
				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(k)
					break
				time.sleep(settings.CLIENT_SLEEP)
			data["success"] = True
	return flask.jsonify(data)

@app.route("/predictSports", methods=["POST"])
def predictSports():
	data = {"success": False}
	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		imgdata = base64.b64decode(data1)
		print(type(imgdata))
		if type(imgdata) is not None:
			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')
			print(type(image))
			image = prepare_image(image,
				(500, 500))
			image = image.copy(order="C")
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'Sports'}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			# predictions
			while True:
				output = db.get(k)
				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(k)
					break
				time.sleep(settings.CLIENT_SLEEP)
			data["success"] = True
	return flask.jsonify(data)

@app.route("/predictHome", methods=["POST"])
def predictHome():
	data = {"success": False}
	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		imgdata = base64.b64decode(data1)
		print(type(imgdata))
		if type(imgdata) is not None:
			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')
			print(type(image))
			image = prepare_image(image,
				(500, 500))
			image = image.copy(order="C")
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'Home'}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			# predictions
			while True:
				output = db.get(k)
				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(k)
					break
				time.sleep(settings.CLIENT_SLEEP)
			data["success"] = True
	return flask.jsonify(data)

@app.route("/predictElectronics", methods=["POST"])
def predictElectronics():
	data = {"success": False}
	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		imgdata = base64.b64decode(data1)
		print(type(imgdata))
		if type(imgdata) is not None:
			image = Image.open(io.BytesIO(imgdata))
			image.save('raw.png')
			print(type(image))
			image = prepare_image(image,
				(500, 500))
			image = image.copy(order="C")
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'Electronics'}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
			# predictions
			while True:
				output = db.get(k)
				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)
					db.delete(k)
					break
				time.sleep(settings.CLIENT_SLEEP)
			data["success"] = True
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
		# data_decode = unquote(data1)
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
			d = {"id": k, "image": image, 'type':'general'}
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

@app.route("/predictTag", methods=["POST"])
def predictTag():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint

	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		# data_decode = unquote(data1)
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
			# image.save('raw.png')

			print(type(image))
			tags = predictTaghelper(image)
			image = prepare_image(image,
				(500, 500))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image, 'type':'general'}
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
					data["tags"] = tags
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


@app.route("/selectTag", methods=["POST"])
def selectTag():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint

	if flask.request.method == "POST":
		print("getting data")
		data1=flask.request.get_data()
		# data_decode = unquote(data1)
		data1 = json.loads(data1)
		# print("json_re is" , json_re)
		#
		# print(type(json_re))
		print(data1)
		asinList = []
		for item in data1["predictions"]:
			asinList.append(item['ASIN'])
		textSearch = ""
		for item in data1["tags"]:
			textSearch=textSearch+" "+item

		print(asinList,textSearch)

		textdata = mycol.find({"$text": {"$search": textSearch}, "ASIN": {
			"$in": asinList}})
		result = []
		for doc in textdata:
			del doc['_id']
			result.append(doc)
		print("result is----------------------",result)
		data["predictions"] = result
		data["tags"] = data1["tags"]
		data["success"] = True
	return flask.jsonify(data)
# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production

# @app.route("/predict", methods=["POST"])
# def predict():
# 	# initialize the data dictionary that will be returned from the
# 	# view
# 	data = {"success": False}
#
# 	# ensure an image was properly uploaded to our endpoint
#
# 	if flask.request.method == "POST":
# 		print("getting data")
# 		data1=flask.request.get_data()
# 		data_decode = unquote(data1)
# 		json_re = json.loads(data_decode)
# 		print("json_re is" , json_re)
#
# 		print(type(json_re))
# 		imgdata = base64.b64decode(data1)
# 		print(type(imgdata))
#
# 		if type(imgdata) is str:
# 			# read the image in PIL format and prepare it for
# 			# classification
#
#
#
# 			image = Image.open(io.BytesIO(imgdata))
# 			image.save('raw.png')
#
# 			print(type(image))
#
# 			image = prepare_image(image,
# 				(500, 500))
#
# 			# ensure our NumPy array is C-contiguous as well,
# 			# otherwise we won't be able to serialize it
# 			image = image.copy(order="C")
#
# 			# generate an ID for the classification then add the
# 			# classification ID + image to the queue
# 			k = str(uuid.uuid4())
# 			image = helpers.base64_encode_image(image)
# 			d = {"id": k, "image": image}
# 			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
#
# 			# keep looping until our model server returns the output
# 			# predictions
# 			while True:
# 				# attempt to grab the output predictions
# 				output = db.get(k)
#
# 				# check to see if our model has classified the input
# 				# image
# 				if output is not None:
# 					# add the output predictions to our data
# 					# dictionary so we can return it to the client
# 					output = output.decode("utf-8")
# 					# data["predictions"] = json.loads(output)
# 					data["predictions"] = json.loads(output)
#
# 					# delete the result from the database and break
# 					# from the polling loop
# 					db.delete(k)
# 					break
#
# 				# sleep for a small amount to give the model a chance
# 				# to classify the input image
# 				time.sleep(settings.CLIENT_SLEEP)
#
# 			# indicate that the request was a success
# 			data["success"] = True
#
# 	# return the data dictionary as a JSON response
# 	return flask.jsonify(data)

if __name__ == "__main__":
	print("* Starting web service...")
	app.run()