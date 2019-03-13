import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import multiprocessing
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import wget
import imghdr
import json
import pickle
import hnswlib
import numpy as np
import glob, os, time
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import urllib.parse
import urllib
import gzip
import os
import tempfile
import glob
from os.path import join
subset_num = 100000

data_path = 'metadata.json'
images_path = join('/data','amazon_images_subset')



if not os.path.isdir(images_path):
    os.makedirs(images_path)
	
num_lines = 0	
num_lines = sum(1 for line in open(data_path))
assert num_lines >= subset_num, "Subset needs to be smaller or equal to total number of example"

def parse(path, num_cpu, modulo):
    g = open(path, 'r')
    for i, l in enumerate(g):
        if (i >= num_lines - subset_num and i%num_cpu == modulo):
            yield eval(l)
def download_files(modulo):
    for i, data in enumerate(parse(data_path, NUM_CPU, modulo)):
        if (i%1000000==0):
            print(i)
        if 'imUrl' in data and data['imUrl'] is not None and 'categories' in data and data['imUrl'].split('.')[-1] == 'jpg':
            url = data['imUrl']
            try:
                path = os.path.join(images_path, data['asin']+'.jpg')
                if not os.path.isfile(path):
                    file = urllib.request.urlretrieve(url, path)
            except:
                print("Error downloading {}".format(url))
				
				
NUM_CPU = multiprocessing.cpu_count()*2
BATCH_SIZE = 256

EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])

# ctx = mx.gpu() if len(mx.test_utils.list_gpus()) else mx.cpu()
#
# net = vision.resnet18_v2(pretrained=True, ctx=ctx).features
#
# net.hybridize()
#
# net(mx.nd.ones((1,3,224,224), ctx=ctx))
# net.export(join('mms','visualsearch'))


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

idx_ASIN = []
for file in list_files:
    idx_ASIN.append(file.split('\\')[-1].split('.')[0])
print(idx_ASIN)
ASINs = set(idx_ASIN)    

def get_ASIN_data(modulo):
    ASIN_data = {}
    file = open(data_path, 'r')
    for i, line in enumerate(file):
        if i % NUM_CPU == modulo:
            data = eval(line)
            if i % 1000000 == 0:
                print("[{}] product data processed".format(i))
            if data['asin'] in ASINs:
                ASIN_data[data['asin']] = {
                    'price':data['price'] if 'price' in data else 'NA',
                    'url':data['imUrl'],
                    'title': data['title'] if 'title' in data else 'NA',
                    'ASIN':data['asin']
                }
    return ASIN_data




if __name__ == '__main__':

    NUM_CPU = 6
    pool = multiprocessing.Pool(processes=NUM_CPU)
    results = pool.map(get_ASIN_data, list(range(NUM_CPU)))
    ASIN_data = { k: v for d in results for k, v in d.items() }
    idx_asin_file = join('mms', 'idx_ASIN.pkl')
    ASIN_data_file = join('mms', 'ASIN_data.pkl')
    pickle.dump(idx_ASIN, open(idx_asin_file, 'wb'), protocol=2)
    pickle.dump(ASIN_data, open(ASIN_data_file, 'wb'), protocol=2)
