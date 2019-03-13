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


if __name__ == '__main__':		

	pool = multiprocessing.Pool(processes=NUM_CPU)

	results = pool.map(download_files, list(range(NUM_CPU)))

	# Removing all the fake jpegs
	list_files = glob.glob(os.path.join(images_path, '**.jpg'))
	for file in list_files:
		if imghdr.what(file) != 'jpeg':
			print('Removed {} it is a {}'.format(file, imghdr.what(file)))
			os.remove(file)