#!/usr/bin/python

"""Estimateur : 2 random forests pour registered et casual

"""

from sklearn.ensemble import *
import numpy

from sklearn.base import BaseEstimator, RegressorMixin

class myEstimator(BaseEstimator, RegressorMixin):
    """Estimateur : count=registered+casual"""

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """Called when initializing the estimator"""
        self.intValue = intValue
        self.stringParam = stringParam
        self.differentParam = otherParam

        self.model_casual = RandomForestRegressor(
        n_estimators=500, criterion='mse', max_features='auto', max_depth=None, \
        min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=None, n_jobs=-1, random_state=316)

        self.model_registered = RandomForestRegressor(
        n_estimators=500, criterion='mse', max_features='auto', max_depth=None, \
        min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=None, n_jobs=-1, random_state=316)

        self.casual_features = []
        self.registered_features = []

    def fit(self, X, y=None):
        """Train the two random forests"""

        assert (type(self.intValue) == int), "intValue parameter must be integer"
        assert (type(self.stringParam) == str), "stringValue parameter must be string"

        self.model_registered.fit(X, y[:,2])
        self.model_casual.fit(X, y[:,1])

        self.feature_importances_ =  self.model_registered.feature_importances_
        self.estimators_ = self.model_registered.estimators_

        return self

    def predict(self, X, y=None):
        """Predict registered and casual then sum"""
        try:
            getattr(self, "model_casual")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        yregistered = self.model_registered.predict(X)
        ycasual = self.model_casual.predict(X)

        return  yregistered + ycasual

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))