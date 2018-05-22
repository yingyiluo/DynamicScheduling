#!/usr/bin/python3

import numpy as np
import numpy.random as random
import sklearn.ensemble as ensemble
import matplotlib.pyplot as plt
from sklearn import datasets

class GradientBoosting():
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
        self.model = ensemble.GradientBoostingRegressor(**args)

    def default(self):
        # Good # 1
        params = {
            "loss": "quantile",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "subsample": 1.,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "alpha": 0.9,
            "init": None,
            "verbose": 0,
            "warm_start": False,
            "random_state": None,
            "presort": "auto",
        }
        params = {
            "loss": "ls",
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.,
            "subsample": 1.,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "alpha": 0.9,
            "init": None,
            "verbose": 0,
            "warm_start": False,
            "random_state": None,
            "presort": "auto",
        }
        self.args = params
        self.models = []
        self.model = ensemble.GradientBoostingRegressor(**params)

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
        feature_names = np.array(['cyc_1', 'cyc_1', 'inst_1', 'inst_1', 'llcref_1', 'llcref_1', 'llcmiss_1', 'llcmiss_1', 'br_1', 'br_1', 'brm_1', 'brm_1', 'cyc_0', 'cyc_1_1', 'inst_0', 'inst_1_1', 'llcref_0', 'llcref_1_1', 'llcmiss_0', 'llcmiss_1_1', 'br_0', 'br_1_1', 'brm_0', 'brm_1_1', 'l2lin_1_1', 'dtemp_0', 'tdie_0', 'avgpwr_0', 'FanGroup1_0_1', 'FanGroup3_0_1', 'Fan1A_0_1', 'Fan1B_0_1', 'Fan2A_0_1', 'Fan2B_0_1', 'Fan3A_0_1', 'Fan3B_0_1', 'Fan4A_0_1', 'Fan4B_0_1', 'Fan5A_0_1', 'Fan5B_0_1', 'Fan6A_0_1', 'Fan6B_0_1', 'Fan7A_0_1', 'Fan7B_0_1', 'dtemp_1_1', 'tpkg_1_1', 'power_1_1'])
        print(self.args, self.X.shape, self.y.shape)
        X = self.X
        y = self.y
        if self.m is not None and self.m < len(self.X):
            sels = random.choice(X.shape[0], self.m, replace = False)
            X = X[sels]
            y = y[sels]
        self.nys = y.shape[1]
        for i in range(self.nys):
            self.models.append(ensemble.GradientBoostingRegressor(**self.args))
            self.models[i].fit(X, y[:,i])
            #if i == 1:
                #print('find column for dtemp0')
                #feature_importance = self.models[i].feature_importances_
                #feature_importance = 100.0 * (feature_importance / feature_importance.max())
                #feature_importance = np.delete(feature_importance, [12, 13, 26])
                #sorted_idx = np.argsort(feature_importance)
                #sorted_idx = sorted_idx[-10:]
                #pos = np.arange(sorted_idx.shape[0]) + .5
                #plt.figure()
                #plt.barh(pos, feature_importance[sorted_idx], align='center')
                
                #print(sorted_idx)
                #plt.yticks(pos, feature_names[sorted_idx])
                #plt.xlabel('Relative Importance')
                #plt.title('Feature Importance for Temperature Prediction')
                #plt.savefig("temp_10r.eps", bbox_inches='tight')
                #print(feature_importance)
            if i == 2:
                feature_importance = self.models[i].feature_importances_
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                feature_importance = np.delete(feature_importance, [12, 13, 26])
                sorted_idx = np.argsort(feature_importance)
                sorted_idx = sorted_idx[-10:]
                pos = np.arange(sorted_idx.shape[0]) + .5
                plt.figure()
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                
                print(sorted_idx)
                plt.yticks(pos, feature_names[sorted_idx])
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance for Power Prediction')
                plt.savefig("power_10r.eps", bbox_inches='tight')
                print(feature_importance)
                
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
