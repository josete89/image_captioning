
from data_processing import *








if __name__ == '__main__':
    print "Reading data"
    imag_traing = readPhotos()
    text_train = readFile()
    print text_train
    print "Preprocesing data"