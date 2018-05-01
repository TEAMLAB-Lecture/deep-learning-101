import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

import os

dataset_dir = "notMNIST_large"

classes = []
labels = []
features = []

for root, dirs, files in os.walk(dataset_dir, topdown=False):
    for name in dirs:
        classes.append(name)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(classes)
print(le.classes_)


for class_name in classes:
    root_dir = os.path.join(dataset_dir, class_name)
    for name in os.listdir(root_dir):
        img_path = os.path.join(root_dir, name)
        try:
            img_ndarray = mpimg.imread(img_path)
            features.append(img_ndarray.flatten())
            labels.append(class_name)
        except OSError as e:
            print(img_path)
            print(e)
        except ValueError as e:
            print(img_path)
            print(e)

le = preprocessing.LabelEncoder()
le.fit(classes)
labels_int = le.transform(labels)
ohe = preprocessing.OneHotEncoder()
ohe.fit(labels_int.reshape(-1, 1))
labels_one_hot = ohe.transform(labels_int.reshape(-1, 1))

data = {}
data["features"] = np.asarray(features)
data["labels"] = np.asarray(labels_one_hot.toarray())

train_set = {}
test_set = {}


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

for train_index, test_index in sss.split(data["features"], data["labels"]):
    train_set["features"] = data["features"][train_index]
    train_set["labels"] = data["labels"][train_index]

    test_set["features"] = data["features"][test_index]
    test_set["labels"] = data["labels"][test_index]

# for MAC
from zodbpickle import pickle # for Mac OS

with open('notMNIST_large.npy', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('notMNIST_large_train.npy', 'wb') as f:
    pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('notMNIST_large_test.npy', 'wb') as f:
    pickle.dump(test_set, f, protocol=pickle.HIGHEST_PROTOCOL)

# for Linux
# np.save("notMNIST_large.npy", data)
# np.save("notMNIST_large_train.npy", train_set)
# np.save("notMNIST_large_test.npy", test_set)
