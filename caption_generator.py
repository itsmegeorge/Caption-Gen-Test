"""Caption Generator using Keras and VGG19.
Walkthrough provided mainly by Jason Browlee,"""

"""this script uses the generated tokenizer and model to generate
captions for new, unseen images"""

"""Todo: Replace VGG16 with InceptionV3 after building model with it. Need to find the proper hyperparameters and also use MSCOCO for training"""



import keras
from numpy import argmax
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from pickle import dump
from pickle import load
from keras.preprocessing.sequence import pad_sequences

def extract_features(filename):
	model = VGG16() #load model
	model.layers.pop() #split the layers
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output) #I only want the last one so to know what each image is about
	picture=load_img(filename, target_size=(224,224))
	picture=img_to_array(picture) #convert to array
	picture=picture.reshape((1, picture.shape[0], picture.shape[1], picture.shape[2])) #reshape tensors
	picture=preprocess_input(picture) #ready image for model
	feature = model.predict(picture, verbose=1) #get feature
	return feature

picture=extract_features('example.jpg')

def integer_to_word(integer,tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'START' #start generating. String gives 0th word to the model
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0] #turn sequence to integers
		sequence = pad_sequences([sequence], maxlen=max_length) #pad sequence
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0) #predict next word
		yhat = argmax(yhat) #turn probability to integer 
		word = integer_to_word(yhat, tokenizer) #match integer to word
		if word is None: # stop if cannot match
			break
		in_text += ' ' + word #space between previous and next word
		if word == 'END' or word == 'end': #stop generating. Also why is this case sensitive and needs both?
			break
	return in_text[6:-3] #remove START and END. Hacky but take it for now
 
tokenizer = load(open('tokenizer.pkl', 'rb')) #load tokenizer
max_length = 34
model = load_model('model-ep005-loss3.552-val_loss3.845.h5') #load model
photo = extract_features('example.jpg') #load image extract feature
description = generate_desc(model, tokenizer, photo, max_length) #tell me what you see!
print(description)
