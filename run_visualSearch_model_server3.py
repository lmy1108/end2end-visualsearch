# # import the necessary packages
# from keras.applications import ResNet50
# from keras.applications import imagenet_utils
import numpy as np
import settings
import helpers
import redis
import time
import json
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import multiprocessing
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import imghdr
import json
import pickle
import hnswlib
import numpy as np
import glob, os, time
import urllib
import gzip
import os
import tempfile
import glob
# from download import download_files
from os.path import join
import urllib.parse
import base64

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

data_path = 'metadata.json'
images_path = join('d:\\data','Clothing,Shoes&Jewelry')
BATCH_SIZE = 256
EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])
ctx = mx.cpu()
net = vision.resnet18_v2(pretrained=True, ctx=ctx).features
net.hybridize()

net(mx.nd.ones((1,3,224,224), ctx=ctx))
net.export(join('mms','visualsearch'))

def transform(image, label):
    resized = mx.image.resize_short(image, SIZE[0]).astype('float32')
    cropped, crop_info = mx.image.center_crop(resized, SIZE)
    cropped /= 255.
    normalized = mx.image.color_normalize(cropped,
                                      mean=MEAN_IMAGE,
                                      std=STD_IMAGE) 
    transposed = nd.transpose(normalized, (2,0,1))
    return transposed, label

empty_folder = tempfile.mkdtemp()
# Create an empty image Folder Data Set
dataset = ImageFolderDataset(root=empty_folder, transform=transform)

list_files = glob.glob(os.path.join(images_path, '**.jpg'))

print("[{}] images".format(len(list_files)))

idx_asin_file = join('mms','idx_ASIN.pkl')
ASIN_data_file = join('mms', 'ASIN_data.pkl')

idx_ASIN = pickle.load(open(idx_asin_file, 'rb'))
Clothingidx_ASIN = pickle.load(open(join('mms','Clothing,Shoes&Jewelryidx_ASIN.pkl'), 'rb'))
Booksidx_ASIN = pickle.load(open(join('mms','Booksidx_ASIN.pkl'), 'rb'))
Sportsidx_ASIN = pickle.load(open(join('mms','Sports&Outdoorsidx_ASIN.pkl'), 'rb'))
Homeidx_ASIN = pickle.load(open(join('mms','Home&Kitchenidx_ASIN.pkl'), 'rb'))
Electronicsidx_ASIN = pickle.load(open(join('mms','Electronicsidx_ASIN.pkl'), 'rb'))



ASIN_data = pickle.load(open(ASIN_data_file, 'rb'))

dataset.items = list(zip(list_files, [0]*len(list_files)))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, last_batch='keep', shuffle=False, num_workers=multiprocessing.cpu_count())

p = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE) # possible options are l2, cosine or ip
pBooks = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE)
pClothing = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE)
pElectronics = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE)
pHome = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE)
pSports = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE)
# # Initing index - the maximum number of elements should be known beforehand
# p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)
#
# # Element insertion (can be called several times):
# int_labels = p.add_items(features, labels_index)


p.load_index(join('mms','index.idx'))
pBooks.load_index(join('mms','Booksindex.idx'))
pClothing.load_index(join('mms','Clothing,Shoes&Jewelryindex.idx'))
pElectronics.load_index(join('mms','Electronicsindex.idx'))
pHome.load_index(join('mms','Home&Kitchenindex.idx'))
pSports.load_index(join('mms','Sports&Outdoorsindex.idx'))



feature_file = join('mms','Clothing,Shoes&Jewelryfeature_data.pkl')

features = pickle.load(open(feature_file, 'rb'))

def plot_predictions(images):
    gs = gridspec.GridSpec(3, 3)
    gs.update(hspace=0.1, wspace=0.1)
    for i, (gg, image) in enumerate(zip(gs, images)):
        gg2 = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=gg)

def search(N, k):
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    q_labels, q_distances = p.knn_query([features[N]], k = k+1)
    images = [plt.imread(dataset.items[label][0]) for label in q_labels[0][1:]]
    labels_find = [(dataset.items[label]) for label in q_labels[0][1:]]

    print(images)
    plot_predictions(images)


# index = np.random.randint(0,len(features))
k = 10
# search(122, k)
print("search succeeded!")

def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	# print("* Loading model...")
	# model = ResNet50(weights="imagenet")
	# print("* Model loaded")

    # continually pool for new images to classify
	while True:

            buffer = db.lrange(settings.IMAGE_QUEUE, 0 , 0)

            if len(buffer) != 0:
                curImage = json.loads(buffer[0].decode('utf-8'))
                print(type(curImage),type(curImage))
                image = helpers.base64_decode_image(curImage["image"],
                    settings.IMAGE_DTYPE,
                    (500, 500,
                    settings.IMAGE_CHANS))
                imageId = curImage['id']

                print(type(image),image.shape)

                image_t, _ = transform(nd.array(image), 1)


                output = net(image_t.expand_dims(axis=0).as_in_context(ctx))

                type1 = curImage['type']
                if(type1 == 'general'):
                    labels, distances = p.knn_query([output.asnumpy().reshape(-1, )], k=30)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = idx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))


                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)

                elif(type1 == 'books'):
                    labels, distances = pBooks.knn_query([output.asnumpy().reshape(-1, )], k=35)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = Booksidx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))

                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)

                elif (type1 == 'Clothing'):
                    labels, distances = pClothing.knn_query([output.asnumpy().reshape(-1, )], k=35)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = Clothingidx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))

                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)

                elif (type1 == 'Sports'):
                    labels, distances = pSports.knn_query([output.asnumpy().reshape(-1, )], k=35)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = Sportsidx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))

                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)
                elif (type1 == 'Home'):
                    labels, distances = pHome.knn_query([output.asnumpy().reshape(-1, )], k=35)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = Homeidx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))

                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)
                elif (type1 == 'Electronics'):
                    labels, distances = pElectronics.knn_query([output.asnumpy().reshape(-1, )], k=35)

                    print("knn succeeded!")

                    output = []
                    for label in labels[0]:
                        print(label, "  ")
                        ASIN = Electronicsidx_ASIN[label]
                        if ASIN in ASIN_data:
                            output.append(ASIN_data[ASIN])
                    db.set(imageId, json.dumps(output))

                    print("saved in redis.")
                    db.ltrim(settings.IMAGE_QUEUE, len(curImage), -1)

            time.sleep(settings.SERVER_SLEEP)
        #
		# imageIDs = []
		# batch = None
        #
		# # loop over the queue
		# for q in queue:
		# 	# deserialize the object and obtain the input image
		# 	q = json.loads(q.decode("utf-8"))
		# 	image = helpers.base64_decode_image(q["image"],
		# 		settings.IMAGE_DTYPE,
		# 		(1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
		# 			settings.IMAGE_CHANS))
        #
		# 	# check to see if the batch list is None
		# 	if batch is None:
		# 		batch = image
        #
		# 	# otherwise, stack the data
		# 	else:
		# 		batch = np.vstack([batch, image])
        #
		# 	# update the list of image IDs
		# 	imageIDs.append(q["id"])
        #
		# # check to see if we need to process the batch
		# if len(imageIDs) > 0:
		# 	# classify the batch
		# 	print("* Batch size: {}".format(batch.shape))
		# 	preds = model.predict(batch)
        #
        #
        #     image_t, _ = transform(nd.array(image), 1)
        #     output = net(image_t.expand_dims(axis=0).as_in_context(ctx))
        #     labels, distances = p.knn_query([output.asnumpy().reshape(-1, )], k=6)
        #
        #
        #
		# 	results = imagenet_utils.decode_predictions(preds)
        #
		# 	# loop over the image IDs and their corresponding set of
		# 	# results from our model
		# 	for (imageID, resultSet) in zip(imageIDs, results):
		# 		# initialize the list of output predictions
		# 		output = []
        #
		# 		# loop over the results and add them to the list of
		# 		# output predictions
		# 		for (imagenetID, label, prob) in resultSet:
		# 			r = {"label": label, "probability": float(prob)}
		# 			output.append(r)
        #
		# 		# store the output predictions in the database, using
		# 		# the image ID as the key so we can fetch the results
		# 		db.set(imageID, json.dumps(output))
        #
		# 	# remove the set of images from our queue
		# 	db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)
        #
		# # sleep for a small amount
		# time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	classify_process()