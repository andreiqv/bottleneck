#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip
import random
import numpy as np
np.set_printoptions(precision=4, suppress=True)
#import tensorflow_hub as hub


def load_data(in_dir, img_size=(299,299)):	
	""" each image has form [height, width, 3]
	"""

	data = dict()
	data['filenames'] = []
	data['images'] = []
	data['labels'] = []

	files = os.listdir(in_dir)
	random.shuffle(files)

	for file_name in files:

		file_path = in_dir + '/' + file_name

		#img_gray = Image.open(file_path).convert('L')
		#img = img_gray.resize(img_size, Image.ANTIALIAS)
		img = Image.open(file_path)
		img = img.resize(img_size, Image.ANTIALIAS)
		arr = np.array(img, dtype=np.float32) / 256

		name = ''.join(file_name.split('.')[:-1])
		angle = name.split('_')[-1]
		lable = float(angle) / 360.0

		if type(lable)!=float:
			print('type(lable)!=float')
			raise Exception('lable type is not float')

		print('{0}: {1:.3f}, {2}' .format(angle, lable, file_name))
		data['images'].append(arr)
		data['labels'].append(lable)
		data['filenames'].append(file_name)


	return data
	#return train, valid, test

"""
def covert_data_to_feature_vector(data):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
	height, width = hub.get_expected_image_size(module)
	assert (height, width) == (299, 299)
	image_feature_vector = module(data['images'])
	data['images'] = image_feature_vector
	return data
"""

def split_data(data, ratio=(6,1,3)):

	len_data = len(data['images'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)

	print(len_train, len_valid, len_test)

	data_train = dict()
	data_valid = dict()
	data_test = dict()

	data_train['images'] = data['images'][ : len_train]
	data_train['labels'] = data['labels'][ : len_train]
	data_train['filenames'] = data['filenames'][ : len_train]

	data_valid['images'] = data['images'][len_train : len_train + len_valid]
	data_valid['labels'] = data['labels'][len_train : len_train + len_valid]
	data_valid['filenames'] = data['filenames'][len_train : len_train + len_valid]

	data_test['images'] = data['images'][len_train + len_valid : ]
	data_test['labels'] = data['labels'][len_train + len_valid : ]
	data_test['filenames'] = data['filenames'][len_train + len_valid : ]
  
	data_train['size'] = len(data_train['images'])
	data_valid['size'] = len(data_valid['images'])
	data_test['size'] = len(data_test['images'])

	splited_data = {'train': data_train, 'valid': data_valid, 'test': data_test}

	return splited_data


if __name__ == '__main__':

	in_dir = 'data'
	data1 = load_data(in_dir, img_size=(299,299))

	print(len(data1['images']))
	print(len(data1['labels']))

	#data2 = covert_data_to_feature_vector(data1)

	data = split_data(data1, ratio=(6,1,3))

	print('train', data['train']['size'])
	print('valid', data['valid']['size'])
	print('test',  data['test']['size'])

	# add_pickle

	dump = pickle.dumps(data)
	print('dump done')
	f = gzip.open('dump.gz', 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()