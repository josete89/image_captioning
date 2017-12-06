from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from numpy import argmax, array

from data_processing import *
from model import *
from coremltools.converters.keras import convert
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pickle import load
from keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model

from pickle import dump


def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


def load_text(textData, dataset):
    features = {k: textData[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        all_desc.append(descriptions[key])
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def create_sequences(tokenizer, max_length, descriptions, photos,vocab_size):
    X1, X2, y = list(), list(), list()
    for key in descriptions:
        desc = descriptions[key]
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photos[key][0])
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

def convertToCoreML(model):
    coreml_model = convert(model)
    coreml_model.save('shoesGenerator.mlmodel')





# CHECK MODEL ACCURACY
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


if __name__ == '__main__':
    print "--- READING DATA -----"

    text = readFile()

    print "--Preprocesing data--"
    photosId = listOfPhotoId("./../data/")

    bag_of_lines = preprocessText(text)
    counter = getWordsCount(bag_of_lines)
    set_of_words = getUniqueWords(bag_of_lines)
    vocab_size = len(set_of_words)
    max_caption_length = maxLengthCaption(bag_of_lines)
    textData = textDataSet(text)

    rangeOfTrainingSet = int(round(float(len(photosId)) * float(0.8)))
    rangeOfTestSet = len(photosId) - rangeOfTrainingSet

    trainIds = photosId[0:rangeOfTrainingSet]
    testIds = photosId[rangeOfTrainingSet:]

    print "--- Creating train set ---"
    train_features_images = load_photo_features('features.pkl', trainIds)
    train_description = load_text(textData, trainIds)
    tokenizer = create_tokenizer(train_description)
    dump(tokenizer, open('tokenizer.pkl', 'wb'))
    # prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_caption_length, train_description, train_features_images,vocab_size)

    print "--- Creating test set ---"
    test_features_images = load_photo_features('features.pkl', testIds)
    test_description = load_text(textData, testIds)
    X1test, X2test, ytest = create_sequences(tokenizer, max_caption_length, test_description, test_features_images,vocab_size)

    print "--Creating model-"

    model = define_model(vocab_size, max_caption_length)
    filepath = 'model-best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # fit model
    model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


    model = load_model(filepath)
    # evaluate model
    evaluate_model(model, test_description, test_features_images, tokenizer, max_caption_length)
    convertToCoreML(model)
