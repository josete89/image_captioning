
import numpy as np
import cv2
import os
import os.path

import glob

import collections
from sets import Set

def readPhotos():
    img_size = 232
    dir = "./../data/"
    paths = glob.glob(os.path.normpath(os.getcwd() + dir + "*.jpg" ))
    x_train = []
    for path in paths:
        im = cv2.imread(path)
        im.resize((224, 224, 3))
        im = np.asarray(im)
        print im.shape
        x_train.append(im)
    return np.array( x_train )

def readFile(name="final_labels.txt"):
    dir = "./../data/"
    paths = glob.glob(os.path.normpath(os.getcwd() + dir + name))
    if len(paths) == 1:
        file = open(paths[0],"r")
        text_file = file.read()
        return text_file
    else:
        print "Label file not found!"

def listOfPhotoId(directory):
    paths = glob.glob(os.path.normpath(os.getcwd() + directory + "*.jpg"))
    listOfPhotos = list()

    for path in paths:
        image_id = path.split('.')[0]
        image_id = image_id.split('/')[6]
        listOfPhotos.append(image_id)

    return listOfPhotos

def preprocessText(bunchOfText):
    lines = bunchOfText.split("\n")
    lines = list(filter(lambda x:len(x)>1,lines))
    lines = list(map(lambda x:'startseq ' +x.split(":")[1]+ ' endseq',lines))
    return lines

def textDataSet(bunchOfText):
    lines = bunchOfText.split("\n")
    lines = list(filter(lambda x:len(x)>1,lines))
    lines = list(map(lambda x:x,lines))
    data = {x.split(":")[0]:'startseq ' +x.split(":")[1]+ ' endseq' for x in lines }
    return data

def getWordsCount(lines):
    counter = collections.Counter()
    for line in lines:
        for word in line.split(" "):
            counter[word] += 1
    return counter

def getUniqueWords(lines):
    set = Set()
    for line in lines:
        for word in line.split(" "):
            set.add(word)
    return set

def maxLengthCaption(lines):
    chars_count = sorted(list(map(lambda x:len(x.split(" ")),lines)))
    return chars_count[len(chars_count)-1]


