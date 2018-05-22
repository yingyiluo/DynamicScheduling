#!/usr/bin/python3

import numpy as np
import numpy.random as random
from sklearn import svm

class SVM():
    def __init__(self):
        self.X = None
        self.y = None

    def clean(self):
        self.X = None
        self.y = None
        return self

    def init(self, **args):
        self.m = None
        if 'm' in args:
            self.m = args['m']
            del args['m']
        self.args = args
        self.models = []
        # self.SD = libadvgauss.ivmGP(m = m, corr = corr, hyper = hyper, dcorr = dcorr, normalize = normalize, sigma = sigma, theta = theta, regr = regr)
        self.model = svm.SVR(**args)

    def default(self):
        # Good # 1
        params = {
        #    "fit_intercept": True,
        #    "normalize": True,
        }
        self.args = params
        self.models = []
        self.model = svm.SVR(**params)

    def train(self, X, y):
        if self.X is None:
            self.X = X
        else:
            self.X = np.vstack([self.X, X])

        if self.y is None:
            self.y = y
        else:
            self.y = np.vstack([self.y, y])

    def data(self):
        return self.X, self.y

    def fit(self):
        print(self.args, self.X.shape, self.y.shape)
        X = self.X
        y = self.y
        if self.m is not None and self.m < len(self.X):
            sels = random.choice(X.shape[0], self.m, replace = False)
            X = X[sels]
            y = y[sels]
        self.nys = y.shape[1]
        for i in range(self.nys):
            self.models.append(svm.SVR(**self.args))
            self.models[i].fit(X, y[:,i])
            #if i == 0:
                #print('find column for dtemp0')
                #feature_importance = self.models[i].feature_importances_
                #feature_importance = 100.0 * (feature_importance / feature_importance.max())
                #sorted_idx = np.argsort(feature_importance)
                #pos = np.arange(sorted_idx.shape[0]) + .5
                #print(feature_importance)
                
    def predict(self, X):
        ys = None
        for i in range(self.nys):
            if ys is None:
                ys = self.models[i].predict(X)
                if X.ndim == 2:
                    ys = np.reshape(ys, (len(ys), 1))
            else:
                res = self.models[i].predict(X)
                if X.ndim == 1:
                    ys = np.hstack([ys, res])
                else:
                    ys = np.hstack([ys, np.reshape(res, (len(res), 1))])
        return ys
