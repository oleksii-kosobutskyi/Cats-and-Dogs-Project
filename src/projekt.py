import sys
from os import listdir
from os.path import isfile, join
import random

import librosa
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from playsound import playsound

TRAIN_FOLDER = "../data/train/"
TEST_FOLDER = "../data/test/"


def audio2array(file_name):

    # here kaiser_fast is a technique used for faster extraction
    [data, sample_rate] = librosa.load(file_name, res_type='kaiser_fast')

    # we extract mfcc feature from data
    feature = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40), axis=1) #40 values
    if ('cat_' in file_name):
        label = 0
    elif ('dog_' in file_name):
        label = 1
    else:
        label = -1

    return [feature, label]


def train():

    # function to load files and extract features
    X = np.empty((0,40)); Y = np.empty(0)
    file_names = [join(TRAIN_FOLDER, file) for file in listdir(TRAIN_FOLDER) if isfile(join(TRAIN_FOLDER, file))]
    for file_name in file_names:
        # handle exception to check if there isn't a file which is corrupted
        try:
            [x, y] = audio2array(file_name)
        except Exception as exc:
            print(f"Ups, cos sie spieprzylo.\n{exc}")
            continue
        X = np.vstack([X,x]); Y = np.hstack([Y,y])

    Y = to_categorical(Y) # Convert class vector (integers) to binary class matrix

    # build model
    model = Sequential()

    model.add(Dense(256, activation='relu', input_dim=40))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    model.fit(X, Y, batch_size=32, epochs=5)

    # save model and architecture to single file
    model.save("neural_net.h5")


#if __name__ == '__main__':

if (len(sys.argv) != 2):
    print("Podaj nazwe pliku i tylko to, ty frajerze!")
    sys.exit()
else:

    string = input()
    answer_list = ['Nie chce!', 'Popros ladniej.', 'Popros mnie jeszcze raz!']
    while string != 'Szanowny programie, bardzo prosze przetestowac plik':
        pos = random.randint(0,len(answer_list)-1)
        print(answer_list[pos])
        del answer_list[pos]
        string = input()
        
    # load model
    model = load_model('neural_net.h5')

    # load test file
    file_name = join(TEST_FOLDER, sys.argv[1])
    try:
        [x, y] = audio2array(file_name)
    except Exception as exc:
        print(f"Ups, cos sie spieprzylo.\n{exc}")
    if y == 0:
        y = 'kocisko'
    elif y == 1:
        y = 'psisko'
    else:
        y = 'kogo to obchodzi?'

    # evaluate loaded model on test data
    output = np.argmax(model.predict(np.array([x]), batch_size=32))
    if output == 0:
        output = 'kocisko'
    elif output == 1:
        output = 'psisko'

    playsound(file_name)

    print(f"Mysle ze to {output}.\nTak naprawde, jest to... {y}")
