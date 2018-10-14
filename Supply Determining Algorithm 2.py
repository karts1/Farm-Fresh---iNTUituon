from sklearn.base import TransformerMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split

def range_scaler(oldmin, oldmax, newmin, newmax, value):
	oldrange = oldmax-oldmin
	newrange = newmax-newmin
	newvalue = (((value - oldmin)*newrange)/oldrange)+newmin
	return newvalue

my_data = genfromtxt('/Users/samarth/Desktop/data.csv', delimiter=',')


for item in range(0, my_data.shape[0]):
	var = my_data[item][4]
	my_data[item][4] = int(range_scaler(5538, 1200000, 100, 1000, var))


my_data = my_data[np.logical_not(np.logical_and(my_data[:,4] <100, my_data[:,4] >1000))]
my_data = my_data[np.logical_not(my_data[:,4] > my_data[:,6])]


X = my_data[:,:6]
y = my_data[:,6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class RidgeTransformer(Ridge, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X)

def build_model():
    ridge_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly_feats', PolynomialFeatures()),
        ('ridge', RidgeTransformer())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer),
            ('rand_forest', RandomForestTransformer()),
            ('knn', KNeighborsTransformer())
        ],
        n_jobs=2
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model

print('Build and fit a model..')

model = build_model()

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print('Done. Score:', score)
