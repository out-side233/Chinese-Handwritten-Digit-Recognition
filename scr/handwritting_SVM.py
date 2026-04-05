#!/usr/bin/env python
# coding: utf-8
#Using Jupyter notebook

# Import modules
import numpy as np
import glob
from PIL import Image
import random
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import make_scorer,accuracy_score

# Processing images and reading data
path_ = r'D:\桌面\nus\ECE相关\Machine Learning\Handwritten Chinese character recognition\Handwritten-Chinese-Recognition-project\data'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
data = []
for index, name in enumerate(classes):
    path_all = os.path.join(path_, name, '*.png')
    path_all = glob.glob(path_all)
    
    for img_name in path_all:
        img = Image.open(img_name)
        img = img.convert('L')
        img = img.resize((14, 14))
        img = img.point(lambda i: (255-i)//16)
        data_i = [np.array(img).flatten().tolist(), index]
        data.append(data_i)

# Scramble the data
random.shuffle(data)
print(type(data))

# obtain X，Y
data = np.array(data, dtype=object)
img = data[:, 0]
label = data[:, 1]
#Converts read data into a list
img = img.tolist()
label = label.tolist()

#processing the dataset
X = img
Y = label
X=np.array(X)
Y=np.array(Y)

# Data partitioning
split_index = int( 0.8* len(Y))
x_train, x_test = X[:split_index], X[split_index:]
y_train, y_test = Y[:split_index], Y[split_index:]
print(type(x_train), type(y_train))
print("Training data format:{}".format(np.shape(x_train)))
print("Testing data format:{}".format(np.shape(x_test)))

#Obtain validation set
split_index2 = int( 0.8* len(x_train))
x_train, x_val = x_train[:split_index2], x_train[split_index2:] 
y_train, y_val= y_train[:split_index2], y_train[split_index2:]

# Model construction
model_svc = SVC(kernel='rbf')
model_svc.fit(x_train, y_train)

# Model effect
print("Effect on training data:{}".format(model_svc.score(x_train, y_train)))
print("Effect on testing data:{}".format(model_svc.score(x_test, y_test)))

#K fold cross validation(k=5)
scores = cross_val_score(model_svc,X, Y, cv=5, scoring='accuracy')
#Print out the prediction accuracy for 5 times
print(scores)
#Print out an accurate average of 5 predictions
print(scores.mean())

#PCA
#Normalization of data eigenvalues
train_image =X / 255
train_label = Y
from sklearn.decomposition import PCA

# Use PCA several times to determine the final optimal model
def n_components_analysis(n, x_train, y_train, x_val, y_val):
    
    #PCA dimension reduction implementation
    pca = PCA(n_components=n)
    print("Feature dimensionality reduction, the parameter passed is:{}".format(n))
    pca.fit(x_train)
    
    # Dimensionality reduction in training set and verification set
    x_train_pca = pca.transform(x_train)
    x_val_pca = pca.transform(x_val)
    
    # Training with SVC
    print("Training with SVC")
    ss = SVC()
    ss.fit(x_train_pca, y_train)
    
    # Obtain the accuracy result
    accuracy = ss.score(x_val_pca, y_val)
    return accuracy
n_s = np.linspace(0.8, 0.9, num=5)
accuracy = []
times = []

for n in n_s:
    tmp = n_components_analysis(n, x_train, y_train, x_val, y_val)
    accuracy.append(tmp)

print(accuracy)

pca = PCA(n_components=0.85)

pca.fit(x_train)
pca.n_components_
x_train_pca = pca.transform(x_train)
x_val_pca = pca.transform(x_val)

# Training better model, accuracy calculation

svm = SVC()

svm.fit(x_train_pca, y_train)

svm.score(x_val_pca, y_val)

#SVM model optimization by adjusting parameters
#The grid searches for the best parameters
param_grid_svc = [
    {  
        'C':  [1.5,3.0,3,3.1,3.3],
        'gamma': [0.0009,0.001,0.0011],
        'class_weight': ['balanced',None] 
    }
]
score = make_scorer(accuracy_score)
kf = KFold(n_splits=5,shuffle=False)
grid_search_svc = GridSearchCV(SVC(kernel='rbf'),param_grid_svc,scoring=score,cv=kf)
grid_search_svc.fit(x_train , y_train)

params=grid_search_svc.best_params_#Obtain the best parameters

print(grid_search_svc.best_score_)

# Model construction
svc_optimized = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'],class_weight=params['class_weight'])
svc_optimized.fit(x_train, y_train)


#K fold cross validation(k=5)
scores = cross_val_score(svc_optimized,X, Y, cv=5, scoring='accuracy')
#Print out the prediction accuracy for 5 times
print(scores)
print(scores.mean())
