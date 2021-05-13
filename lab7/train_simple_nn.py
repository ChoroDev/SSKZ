# USAGE
# https://www.reg.ru/blog/keras/
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop,Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        try:
                image = cv2.resize(image, (16, 37)).flatten()
        except:
                continue
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(Dense(1024, input_shape=(1776,), activation="sigmoid"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="sigmoid"))
model.add(Dropout(0.3))
model.add(Dense(258, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))


INIT_LR = 0.01
EPOCHS = 500

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
OPTIMIZER = opt
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

score = model.evaluate(testX, testY, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
