# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:33:03 2018

@author: SHIV
"""

import numpy 
from sklearn import svm

filename = 'G:\ram\digit recognition project\mnist_train.csv'
raw_data = open(filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter=",")

x_train=data[0:50000,1:]
y_train=data[0:50000,0:1]
x_test=data[50000:,1:]
y_test=data[50000:,0:1]


lr_model=svm.SVC(kernel = "poly") #three methods poly,rbf,linear all are good methods 
lr_model.fit(x_train,y_train)

pred=lr_model.predict(x_test)

correct=0

for i in range(0,len(y_test)):
    if pred[i]==y_test[i]:
        correct+=1

print 'accuracy is : ',(float(correct) / len(y_test))*100
