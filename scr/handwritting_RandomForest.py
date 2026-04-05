#!/usr/bin/env python
# coding: utf-8

# # Import modules
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import glob
from PIL import Image
import random
import joblib


# Processing images and reading data
path_ = r'D:\桌面\nus\ECE相关\Machine Learning\Handwritten Chinese character recognition\Handwritten-Chinese-Recognition-project\data'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
all = []
for index, name in enumerate(classes):
    path_all = os.path.join(path_, name, '*.png')
    path_all = glob.glob(path_all)
   
    for img_name in path_all:
        img = Image.open(img_name)
        img = img.convert('L')
        img = img.resize((14, 14))
        img = img.point(lambda i: (255-i)//16)
        data_i = [np.array(img).flatten().tolist(), index]
        all.append(data_i)


# shuffling the data
random.shuffle(all)

all = np.array(all, dtype=object)
print(type(all))

# Obtain the features and targets X,Y
img = all[:, 0]
label = all[:, 1]
img = img.tolist()
label = label.tolist()
X = np.array(img)
Y = np.array(label)

# Splitting the data
split_index = 5 * len(Y) // 6
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = Y[:split_index], Y[split_index:]
print(type(x_train), type(y_train))
print("Training data format:{}".format(np.shape(x_train)))
print("Testing data format:{}".format(np.shape(x_test)))

# Model construction
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("Effect on training data:{}".format(model.score(x_train, y_train)))
print("Effect on testing data:{}".format(model.score(x_test, y_test)))


# Model construction
model = RandomForestClassifier(max_depth=12,n_estimators=110)
model.fit(x_train, y_train)

print("Effect on training data:{}".format(model.score(x_train, y_train)))
print("Effect on testing data:{}".format(model.score(x_test, y_test)))


from sklearn.model_selection import cross_val_score # Import K fold cross validation module

#Use the K fold cross validation module
scores = cross_val_score(model,X, Y, cv=5, scoring='accuracy')
 
#Print out the prediction accuracy for 5 times
print(scores)
 
#Print out an accurate average of 5 predictions
print(scores.mean())



#Import the grid search module
from sklearn.model_selection import GridSearchCV
rfr_best = RandomForestClassifier()
params ={'n_estimators':range(120,125,1),
         'max_depth': [ 8, 9, 10]
        }
gs = GridSearchCV(rfr_best, params, cv=4)
gs.fit(x_train,y_train)
 
#Check the optimized configuration of the hyperparameters
print(gs.best_score_)
print(gs.best_params_)







