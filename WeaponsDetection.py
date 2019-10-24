import cv2

import pandas as pd

import numpy as np

import scipy.misc

import os

from random import shuffle

import tensorflow as tf

import matplotlib.image as mpimg

import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression



TRAIN_DIR = '../input/weapons/images'

TEST_DIR='../input/weapons/timages'

IMG_SIZE = 150

LR = 0.0001

MODEL_NAME = 'WeaponsDetection-cnn'

def tester(vid):

    ret = True

    video_cap = cv2.VideoCapture(vid)

    video_cap.set(cv2.CAP_PROP_FPS, 50)

    while (video_cap.isOpened()):

        # Capture frame-by-frame

        ret, im = video_cap.read()

        boxes = []

        preds = []

        maxs = []

        if not ret:

            break

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        ss.setBaseImage(im)

        ss.switchToSelectiveSearchFast()

        rects = ss.process()

        print('Total Number of Region Proposals: {}'.format(len(rects)))

        for i, rect in enumerate(rects):

            if i<=len(rects):

                x, y, w, h = rect

                if w >=270 and w<300 and h>=120 and h<200:

                    boxes.append(rect)

        print('Total Number of suspicious boxs: {}'.format(len(boxes)))

        imOut = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        for i, rect in enumerate(boxes):

            x, y, w, h = rect

            cropped = imOut[y:y+h , x:x+w]

            cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

            cropped = cropped.reshape(IMG_SIZE, IMG_SIZE, 1)

            prediction = model.predict([cropped])[0]

            p = max(prediction)

            preds.append(p)

            if (p == prediction[0]):

                out = "Grenade"

            elif (p == prediction[1]):

                out = "Machine Guns"

            elif (p == prediction[2]):

                out = "Masked Face"

            elif (p == prediction[3]):

                out = "Pistol Hand Guns"

            elif (p == prediction[4]):

                out = "RPG"

            maxs.append(out)

        if len(boxes) !=0:

            ind=preds.index(max(preds))

            MC=maxs[ind]

            #MyRegions[ind]

            cv2.rectangle(imOut, (boxes[ind][0], boxes[ind][1]), (boxes[ind][0] + boxes[ind][2], boxes[ind][1] + boxes[ind][3]), (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow('Weapons Detection', imOut)

            import winsound

            winsound.PlaySound('s.wav', winsound.SND_FILENAME)

            #MC=max(((item, maxs.count(item)) for item in set(maxs)), key=lambda a: a[1])[0]

            print(MC)

            # wait for 'c' to close the application

            if cv2.waitKey(1) & 0xFF == ord('c'):

                break

    video_cap.release()

    cv2.destroyAllWindows()
	return

def create_label(image_name):

    word_label = image_name

    if word_label == 'Grenade' or 'Grenade' in word_label:

        return np.array([1,0,0,0,0])

    elif word_label == 'Machine Guns' or 'Machine Guns' in word_label:

        return np.array([0,1,0,0,0])

    elif word_label == 'Masked Face' or 'Masked Face' in word_label:

        return np.array([0,0,1,0,0])

    elif word_label == 'Pistol Hand Guns' or 'Pistol Hand Guns' in word_label:

        return np.array([0,0,0,1,0])

    elif word_label == 'RPG' or 'RPG' in word_label:

        return np.array([0,0,0,0,1])

    

def create_train_data():

    training_data = []

    folders = glob.glob('../input/weapons/images/images/*')

    for dir in folders:

        w= glob.glob(dir+'/*')

        for i in w:

            try:

                img_data = cv2.imread(i,0)

                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))

                training_data.append([np.array(img_data), create_label(i)])

            except:

                pass

    shuffle(training_data)

    np.save('train_data.npy', training_data)        

    return training_data





def create_test_data():

    testing_data = []

    folder = glob.glob('../input/weapons/timages/*')

    for i in folder:

        f=glob.glob(i+'/*')

        for im in f:

            try:

                img_data = cv2.imread(im,0)

                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))

                testing_data.append([np.array(img_data), create_label(im)])

            except:

                pass

    np.save('test_data.npy', testing_data)

    return testing_data


if (os.path.exists('train_data.npy')):

    train_data =np.load('train_data.npy')

else: # If dataset is not created:

    train_data = create_train_data()

if (os.path.exists('test_data.npy')):

    test_data =np.load('test_data.npy')

else:

    test_data = create_test_data()





train = train_data

test = test_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



y_train = [i[1] for i in train]



X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_test = [i[1] for i in test]



tf.reset_default_graph()

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')



conv1 = conv_2d(conv_input, 32, 5, activation='relu')

pool1 = max_pool_2d(conv1, 5)



conv2 = conv_2d(pool1, 64, 7, activation='relu')

pool2 = max_pool_2d(conv2, 7)



conv3 = conv_2d(pool2, 32, 5, activation='relu')

pool3 = max_pool_2d(conv3, 5)



conv4 = conv_2d(pool3, 64, 7, activation='relu')

pool4 = max_pool_2d(conv4, 7)



conv5 = conv_2d(pool4, 128, 5, activation='relu')

pool5 = max_pool_2d(conv5, 5)



fully_layer = fully_connected(pool5, 1024, activation='relu')

fully_layer = dropout(fully_layer, 0.5)



cnn_layers = fully_connected(fully_layer, 5, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)



if (os.path.exists('modell.tfl.meta')):

    model.load('./modell.tfl')

else:

    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,

              validation_set=({'input': X_test}, {'targets': y_test}),

              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



model.save('modell.tfl')



tester('w.mp4')
