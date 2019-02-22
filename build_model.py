"""Caption Generator using Keras and VGG19.
Walkthrough provided mainly by Jason Browlee,
although all functions have been changed"""

#MUST: Put files and images to be used in the same dir. Not allowed to redestribute
#DOES: Develops deep learning model based on extracted features. Uses VGG16.
#RUNS: For 20 epochs. Usually fits on the 4th of 5th. Might reduce rounds. Each epoch lasts 30 mins


from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from pickle import dump
from pickle import load
import glob
import keras
import re
import string
import pickle

#Code consists of 2 parts

#---PART 1: PREPARE DATA---#

#includes photo and text processing

def load_txt(filename): #load text files
	with open(filename) as file:
		read_file=file.read()
		file.close()
	return read_file

captions=load_txt('Flickr8k.token.txt')

def load_descriptions(captions): #load photo descriptions to use for the model
	photo_info=dict() #each photo has an photo_id and a caption. I need to match these for each photo
	for line in captions.split('\n'):
		descriptions=line.split() #split by whitespace
		if len(line)>=2: #check because captions that are long use two lines in the file
			photo_id, description = descriptions[0], descriptions[1:] #zero index is the id, rest is description
			photo_id=photo_id.split('.')[0] #I need the photo_id number only
			description= ' '.join(description) #turn from a list back to string
		if photo_id not in photo_info:
			photo_info[photo_id]=[]
		photo_info[photo_id].append(description)
	return photo_info

descriptions=load_descriptions(captions)

def text_cleaning(descriptions): #clean the captions. Make them lowercase, remove punctuation
	for key, value in descriptions.items():
		for i in range(len(value)):
			desc=value[i]
			desc=desc.split()
			desc=[re.sub(r"[^a-z]",'',word.lower()) for word in desc]
			desc=[word for word in desc if len(word)>1]
			desc=[word for word in desc if word.isalpha()] #this and the above in one line but messed up
			value[i]=' '.join(desc)

text_cleaning(descriptions)

def make_vocabulary(descriptions): #make vocabulary for model
	vocab=set() #use set for intersection, so words will be one of each
	for key in descriptions.keys():
		for description in descriptions[key]:
			vocab.update(description.split())
	return vocab

vocabulary = make_vocabulary(descriptions)


def save_descriptions(descriptions, filename): #save descriptions to file, one per line
	lines=[]
	for key, desc_list in descriptions.items():
		for description in desc_list:
			lines.append(key + ' ' + description)
	data = '\n'.join(lines)
	with open(filename, 'w') as output:
		output.write(data)
		output.close()

save_descriptions(descriptions, 'descriptions.txt') #save descriptions with identifiers

#-----------------------------------------#

#---PART 2: DEVELOP DEEP LEARNING MODEL---#

#includes loading, defining and training model
#BLEU scores obtained when evaluator.py is run instead

def load_id(filename): #load identifier numbers
	photo_ids=load_txt(filename)
	photo_info=[]
	for line in photo_ids.split('\n'):
		if len(line)>=1: #check because captions that are long use two lines in the file
			photo_id=line.split('.')[0] #I need the photo_id number only
			photo_info.append(photo_id)
	return set(photo_info) #return set to get clean set for training

def load_clean_desc(filename, dataset): #load clean descriptions into memory
	doc = load_txt(filename) #load txt
	descriptions = dict()
	for line in doc.split('\n'): #split by newline
		# split line by white space
		tokens = line.split() #split each line by whitespace
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset: #if image not in dataset not used
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'START ' + ' '.join(image_desc) + ' END' #signal beginning and end of token
			descriptions[image_id].append(desc)
	return descriptions

def load_dataset(filename, dataset): #load dataset. Can be train or dev
	all_features=load(open(filename, 'rb'))
	features = {f: all_features[k] for f in dataset}
	return features

def dict_to_list(descriptions): #turn dictionary to list of strings
	descs=[]
	for key in descriptions.keys():
		for description in descriptions[key]:
			descs.append(description)
	return descs

def fit_tokenizer(descriptions): #use Keras API to prepare text and fit the tokenizer
	captions=dict_to_list(descriptions)
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(captions)
	return tokenizer

def make_seq(tokenizer, max_length, descriptions, photos): #create input-output pairs for training
	X1, X2, y=[],[],[]
	for key, captions in descriptions.items():
		for caption in captions:
			sequence=tokenizer.texts_to_sequences([caption])[0]
			for i in range(1, len(sequence)):
				input_pair, output_pair=sequence[:i], sequence[i]
				input_pair=pad_sequences([input_pair], maxlen=max_length)[0]
				output_pair=to_categorical([output_pair], num_classes=vocab_size)[0]
				X1.append(photos[key][0])
				X2.append(input_pair)
				y.append(output_pair)
	return array(X1), array(X2), array(y)

def longest_desc_length(descriptions):
	captions=dict_to_list(descriptions)
	return max(len(d.split()) for d in captions)


def define_model(vocab_size,max_length): #define model to fit
	input1=Input(shape=(4096,))
	feature1=Dropout(0.5)(input1)
	feature2=Dense(256, activation='relu')(feature1)
	#feature3=RepeatVector(max_length)(feature2)
	input2=Input(shape=(max_length,))
	sequence1=Embedding(vocab_size, 256, mask_zero=True)(input2)
	sequence2=Dropout(0.5)(sequence1)
	sequence3=LSTM(256)(sequence2)
	decoder1=add([feature2, sequence3])
	decoder2=Dense(256, activation='relu')(decoder1)
	outputs=Dense(vocab_size, activation='softmax')(decoder2)
	model = Model(inputs=[input1, input2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam') #Adam learns the fastest and is more stable
	print(model.summary())
	return model


filename = 'Flickr_8k.trainImages.txt' #load training dataset
train = load_id(filename)
print('Dataset:  '+ str(len(train)))
train_descriptions = load_clean_desc('descriptions.txt', train) #load clean descriptions
print('Descriptions: train= ' + str(len(train_descriptions)))
train_features = load_dataset('features.pkl', train) #load extracted features
print('Photos: train= ' + str(len(train_features)))
tokenizer = fit_tokenizer(train_descriptions) #prepare tokenizer
vocab_size = len(tokenizer.word_index) + 1 #+1 because iteration starts at 0
print('Vocabulary Size: ' + str(vocab_size))
max_length = longest_desc_length(train_descriptions) #max sequence length for a fairer distribution
print('Description Length: ' + str(max_length))
X1train, X2train, ytrain = make_seq(tokenizer, max_length, train_descriptions, train_features) #prepare sequences
 

filename = 'Flickr_8k.devImages.txt' #load dev/test dataset
test = load_id(filename)
print('Dataset: ' + str(len(test)))
test_descriptions = load_clean_desc('descriptions.txt', test) #load descriptions
print('Descriptions: test= ' + str(len(test_descriptions)))
test_features = load_dataset('features.pkl', test)
print('Photos: test= ' + str(len(test_features)))
X1test, X2test, ytest = make_seq(tokenizer, max_length, test_descriptions, test_features)
 

model = define_model(vocab_size, max_length) #ready the model for fitting
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5' #save after each epoch. Called each time
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #if val loss improves, new file
model.fit([X1train, X2train], ytrain, epochs=10, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest)) #fit model. fingers crossed!









