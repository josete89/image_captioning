
import numpy as np
from PIL import Image, ImageOps
import os
import os.path

import glob

def readPhotos():
    img_size = 232
    dir = "./../data/"
    paths = glob.glob(os.path.normpath(os.getcwd() + dir + "*.jpg" ))
    x_train = []
    for path in paths:
        print path
        im = Image.open(path)
        im = ImageOps.fit(im, (img_size, img_size), Image.ANTIALIAS)
        im = np.asarray(im)
        x_train.append(im)
    return x_train

def readFile(name="final_labels.txt"):
    dir = "./../data/"
    paths = glob.glob(os.path.normpath(os.getcwd() + dir + name))
    if len(paths) == 1:
        file = open(paths[0],"r")
        text_file = file.read()
        return text_file
    else:
        print "Label file not found!"
