from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import os
import os.path

import glob

def extract_features(directory):
	model = VGG16()

	#remove last layer
	model.layers.pop()

	model = Model(inputs=model.inputs,outputs=model.layers[-1].output)

	print(model.summary())

	features = dict()

	paths = glob.glob(os.path.normpath(os.getcwd() + directory + "*.jpg"))

	for path in paths:

		image_id = path.split('.')[0]
		image_id = image_id.split('/')[6]
		print(image_id)
		image = load_img(path,target_size=(224,224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		features[image_id] = feature

	return features

direct = "./../data/"
features = extract_features(direct)
dump(features, open('features.pkl', 'wb'))