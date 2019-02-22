"""Caption Generator using Keras and VGG19.
Walkthrough provided by Jason Browlee,
although functions have been changed"""

#MUST: Put pictures in the same dir. Not allowed to redistribute so request them
#DOES: Extracts features of photos using VGG16. Best model in comparison to both VGG19 and InceptionV3
#RUNS: For approximately one hour on a modern dual core laptop with 8GB of RAM
#TO-DO: Try and tune using InceptionV3. It is faster in extracting features, but needs a larger dataset to have a good score.


import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import dump
import glob

def get_features():
	model = VGG16() #load model
	model.layers.pop() #split the layers
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output) #I only want the last one so to know what each image is about
	print(model.summary()) #print info about layer density and trainable/non-trainable params
	directory=glob.glob('*.jpg') #load photos using glob
	features = dict()
	for image in directory: #extract features from each photo
		picture=load_img(image, target_size=(224,224))
		picture=img_to_array(picture) #convert to array
		picture=picture.reshape((1, picture.shape[0], picture.shape[1], picture.shape[2])) #reshape tensors
		picture=preprocess_input(picture) #ready image for model
		feature=model.predict(picture, verbose=0) #get feature
		image_id=image.split('.')[0] #split on the dot each line has. Get ID
		features[image_id]=feature
		print(str(image))
	return features
 
features = get_features()
print('Extracted Features: ' + str(len(features)))
dump(features, open('features.pkl', 'wb'))

