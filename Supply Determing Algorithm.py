import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.pipeline import Pipeline
import matplotlib.gridspec as gridspec
import itertools
import sklearn
from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner
from sklearn.metrics import mean_squared_error
import pickle
data_arr = np.zeros((60, 6))


clf1 = Pipeline((
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("sgd_reg", ElasticNet(tol=1)),
        ))
clf2 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
clf3 = DecisionTreeRegressor(max_depth=3)



def range_scaler(oldmin, oldmax, newmin, newmax, value):
	oldrange = oldmax-oldmin
	newrange = newmax-newmin
	newvalue = (((value - oldmin)*newrange)/oldrange)+newmin
	return newvalue

my_data = genfromtxt('/Users/samarth/Desktop/data.csv', delimiter=',')


for item in range(0, my_data.shape[0]):
	var = my_data[item][4]
	my_data[item][4] = int(range_scaler(5538, 600000, 100, 1000, var))

'''	
if my_data[item][6] < 100 or my_data[item][6] > 1000 or (my_data[item][6]>my_data[item][4]):
		my_data = np.delete(my_data, (item), axis = 0)
'''
my_data = my_data[np.logical_not(np.logical_and(my_data[:,4] <100, my_data[:,4] >1000))]
my_data = my_data[np.logical_not(my_data[:,4] > my_data[:,6])]


ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack)

clf_list = [clf1, clf2, clf3, eclf, sclf]
lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']
X = my_data[:,:6]
y = my_data[:,6]




d = {yi : i for i, yi in enumerate(set(y))}
y = np.array([d[yi] for yi in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("X_train: " + str(X_train.shape))
print("X_test: "+ str(X_test.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))


gs = gridspec.GridSpec(2, 3)
fig = plt.figure(figsize=(10, 8))

itt = itertools.product([0, 1, 2], repeat=2)

sclf.fit(X_train, y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)

print(result)





