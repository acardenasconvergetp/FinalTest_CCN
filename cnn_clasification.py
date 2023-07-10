import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 32

# Function to load images
def read_images(dirname):
    path = dirname + os.sep
    images = []
    directories = []
    dircount = []
    prevRoot = ''
    cant = 0
    print("Reading images from", path)

    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant += 1
                filepath = os.path.join(root, filename)
                # image = plt.imread(filepath)
                image = cv2.imread(filepath)
                image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH ))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Normalization function
                image = cv2.normalize(image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
                images.append(image)
                if prevRoot != root:
                    prevRoot = root
                    directories.append(root)
                    dircount.append(cant)
                    cant = 0
    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0] = dircount[0] + 1
    print('Number of directories:', len(directories))
    print("Number of images in each directory:", dircount)
    print('Total number of images in subdirectories:', sum(dircount))

    types = []
    index = 0
    for directory in directories:
        name = directory.split(os.sep)
        print(index, name[len(name) - 1])
        types.append(name[len(name) - 1])
        index = index + 1

    labels = []
    index = 0
    for dir_count in dircount:
        for i in range(dir_count):
            labels.append(types[index])
        index = index + 1

    X = np.array(images, dtype=np.uint8)  # convert from list to numpy array
    y = np.array(labels)
    return X, y

# Create Training and Test Sets
X_train, y_train = read_images(os.path.join(os.getcwd(), 'CarneDataset/train'))
X_test, y_test = read_images(os.path.join(os.getcwd(), 'CarneDataset/test'))

# Reshape Images
X_test = X_test.reshape((X_test.shape[0], IMAGE_HEIGHT * IMAGE_WIDTH * 3))
X_train = X_train.reshape(X_train.shape[0], IMAGE_HEIGHT * IMAGE_WIDTH * 3).astype('float32')
X_train = X_train / 255.0

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
knn_model = KNeighborsClassifier(n_neighbors=3, n_jobs=8)
knn_model.fit(X_train, y_train)
predicted_test = knn_model.predict(X_test)
predicted_train = knn_model.predict(X_train)
correct_test = np.where(predicted_test == y_test)[0]
correct_train = np.where(predicted_train == y_train)[0]

print('Number of correctly predicted test values: %s out of %s' % (len(correct_test), len(y_test)))
print('Number of correctly predicted train values: %s out of %s' % (len(correct_train), len(y_train)))