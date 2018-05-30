#!/usr/bin/pythnon3

import numpy as np
import numpy.random as random
import xgboost as xgb
import libdata
import matplotlib.pyplot as plt

class XGBoost():
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
        if 'num_rounds' in args:
            self.num_rounds = args['num_rounds']
        else:
            self.num_rounds = 2
        self.args = args
        self.models = []
        # self.SD = libadvgauss.ivmGP(m = m, corr = corr, hyper = hyper, dcorr = dcorr, normalize = normalize, sigma = sigma, theta = theta, regr = regr)
        # self.model = ensemble.GradientBoostingRegressor(**args)

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
            "random_state": 0,
            "presort": "auto",
            "seed": 0,
        }
        self.args = params
        self.models = []

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
        #print(self.args, self.X.shape, self.y.shape)
        appFeatures = ["%s_%d_%d" % (x, i, t) for t in [1, 0] for x in libdata.apprates for i in [0,1]]
        phyFeatures = ["%s_0" % (x) for x in libdata.targets]
        feature_names = np.append(appFeatures, phyFeatures)
        X = self.X
        y = self.y
        #print(X)
        if self.m is not None and self.m < len(self.X):
            sels = random.sample(X.shape[0], self.m, replace = False)
            X = X[sels]
            y = y[sels]
        self.nys = y.shape[1]
        print(len(X))
        for i in range(self.nys):
            self.models.append(xgb.XGBModel(**self.args))
            self.models[i].fit(X, y[:,i])
            if False:
                #plot_importance(self.models[i])      
                feature_importance = self.models[i].feature_importances_
                feature_importance = 100.0 * (feature_importance / feature_importance.max())
                sorted_idx = np.argsort(feature_importance)
                sorted_idx = sorted_idx[-10:]
                pos = np.arange(sorted_idx.shape[0]) + .5
                plt.figure()
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                print(sorted_idx)
                plt.yticks(pos, feature_names[sorted_idx])
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance for Fan Power Prediction')
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
