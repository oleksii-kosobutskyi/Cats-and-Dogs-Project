import sys
from os import listdir
from os.path import isfile, join

import librosa
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from playsound import playsound

DATA_FOLDER = "../input/audio-cats-and-dogs/cats_dogs/train/cat/"


def audio2array(file_name):

    # here kaiser_fast is a technique used for faster extraction
    [data, sample_rate] = librosa.load(file_name, res_type='kaiser_fast')

    # we extract mfcc feature from data
    feature = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40), axis=1) #40 values
    if ('cat_' in file_name):
        label = 0
    elif ('dog_' in file_name):
        label = 1

    return [feature, label]


def train():

    # function to load files and extract features
    X = np.empty((0,40)); Y = np.empty(0)
    file_names = [join(DATA_FOLDER, file) for file in listdir(DATA_FOLDER) if isfile(join(DATA_FOLDER, file))]
    for file_name in file_names:
        # handle exception to check if there isn't a file which is corrupted
        try:
            [x, y] = audio2array(file_name)
        except Exception as exc:
            print(f"Error encountered while parsing file\n{exc}")
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


if __name__ == '__main__':

    if (len(sys.argv) != 2):
        print("Provide test file name and only that!")
        sys.exit()
    else:

        # load model
        model = load_model('neural_net.h5')

        # load test file
        file_name = sys.argv[1]
        [x, y] = audio2array(file_name)
        y = 'cat' if y == 0 else 'dog'

        # evaluate loaded model on test data
        output = np.argmax(model.predict(np.array([x]), batch_size=32))
        output = 'cat' if output == 0 else 'dog'

        playsound(file_name)

        print(f"I think it is a {output}.\nIt is {y} in fact.")
