import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Normalization, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Input, Lambda
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import math
from tensorflow.keras import datasets, utils
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from PIL import Image

img = cv2.imread('Datasets/Untitled1.png')
model = load_model('char_model.keras')
mean_net = np.loadtxt('mean.txt', dtype=float)
mean_net = mean_net.reshape(1, 28*28)
std_net = np.loadtxt('standy.txt', dtype=float)
std_net = std_net.reshape(1, 28*28)

labels = pd.read_csv('Datasets/emnist-bymerge-mapping.txt',
                     delimiter=' ', header=None, index_col=0)
labels = labels.squeeze()

label_dict = {}

for index, label in enumerate(labels):
    label_dict[index] = chr(label)

def preprocess(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    print(img.shape)
    # cv2.imshow('img1', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    # cv2.imshow('img2', img)

    img = img.reshape(28, 28, 1)
    print(img.shape)
    # cv2.imshow('img3', img)

    single = img.reshape(1, 28*28)
    print(single.shape)

    single = np.array((single - mean_net) / std_net)
    print(single.shape)

    single = single.reshape(1, 28, 28, 1)
    print(single.shape)

    pred = np.argmax(model.predict(single))

    return label_dict[pred]

drawing = False

def draw_circle(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.rectangle(new, (x - 7, y - 7), (x + 7, y + 7), (242, 242, 242), cv2.FILLED)
        # cv2.circle(new, (x, y), 15, (255, 255, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # cv2.circle(new, (x, y), 15, (255, 255, 255), -1)
            cv2.rectangle(new, (x - 7, y - 7), (x + 7, y + 7), (242, 242, 242), cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Create a black image, a window and bind the function to window
new = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (1):
    cv2.imshow('image', new)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

predo = preprocess(new)
cv2.imshow(str(predo), new)
cv2.waitKey(0)