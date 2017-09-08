#!/usr/bin/python

"""Machine learning for bike sharing prediction

"""

import data_exploration as dataexp
import my_estimator as p2est
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import *
from sklearn.model_selection import *
from math import *


def additional_features(data, feature_names):
    '''

    :param data: DataFrame issue du csv de données
    :param feature_names: list
    :return:
    '''

    # Hypothesis : next day free and prev day free influence count
    new_dates = pd.date_range(data.index[0].__str__(), data.index[len(data) - 1].__str__(), freq='H')
    day_groups = data.groupby(lambda x: x.date())
    tmpdata = data.copy(deep=True)
    tmpdata = tmpdata.reindex(new_dates, fill_value=0)

    tmpdata['workingnextday'] = tmpdata.shift(periods=-24, freq='H', axis=1)['workingday']
    tmpdata['workingnextday'] = tmpdata['workingnextday'].fillna(value=0)
    tmpdata['workingprevday'] = tmpdata.shift(periods=24, freq='H', axis=1)['workingday']
    tmpdata['workingprevday'] = tmpdata['workingprevday'].fillna(value=0)

    data= pd.DataFrame(tmpdata, index=data.index)    # retrieve original dates

    feature_names = feature_names + ['workingprevday']

    return [data, feature_names]


def RMSLE_metric(ground_truth, predictions):
    """ Compute Root Mean Squared Logaritmic Error

    Compute RMLSE as in https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError

    :param ground_truth: numpy.ndarray of expected target values
    :param predictions: numpy.ndarray of predicted target values
    :return: float RMSLE error
    """

    T1 = [(lambda x: log(x + 1))(x) for x in list(ground_truth[:,0])]
    T2 = [(lambda x: log(x + 1))(x) for x in list(predictions)]
    DL = [a_i - b_i for a_i, b_i in zip(T1, T2)]
    DLS = [a_i * b_i for a_i, b_i in zip(DL, DL)]
    error = sqrt(sum(DLS)/len(ground_truth))

    return error


def get_features(data):
    """ Clean data, extract new features and return augmented data

    :param data: DataFrame from csv
    :return: list of data, features and feature names
    """
    [data, feature_names] = dataexp.prepare_data(data)
    [data, feature_names] = additional_features(data, feature_names)

    features = data[feature_names].values

    return [data, features, feature_names]


def get_targets(data, estimator):
    """Extract targets from raw data and return targets

    :param data: row DataFrame from csv
    :param estimator: prediction model
    :return: numpy.ndarray each row correspoding to one sample
    """

    targets = []
    if 'count' in data.columns: # train set
        if estimator=="my_estimator":
            targets_names = ['count', 'casual', 'registered']
        else:
            targets_names = ['count']

        targets = data[targets_names].values

    return targets


def predict(ml_procedure, train_path, test_path='data/test.csv', result_path='data/results.csv', \
          cv_sampling='cv_timeseries', estimator='my_estimator', verbose=0):
    """Execute le machine learning et enregistre ou affiche scores et résultats.

    :param ml_procedure: type de procédure de machine learning parmi :
        'crossvalidate' : train sur k-1 fold et test sur le dernier
        'train' : training sur l'ensemble des données du csv de train
        'test' : test sur le csv de test et enregistre les résultats
        'trainandtest' : train + test + enregistre les résultats
        'featureimportance' : affichage de l'importance des features du modèle RF
    :param train_path: chemin vers le csv de données train
    :param test_path: chemin vers le csv de données de test
    :param result_path: chemin vers le csv de sousmission finale
    :param cv_sampling: type d'échantillonage pour le crossvalidation parmi :
        'cv_random' : échantillonage aléatoire
        'cv_timeseries' : échant
    :param estimator: random_forests ou my_estimator
    :param verbose: output detailed results for optimization
    :return:
    """

    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    # See : https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError
    RMSLE_scorer = metrics.make_scorer(RMSLE_metric)


    if estimator == "my_estimator":
        model = p2est.myEstimator()
    elif estimator == "random_forests":
        model = RandomForestRegressor(
        n_estimators=500, criterion='mse', max_features='auto', max_depth=None, \
        min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=None, n_jobs=-1, random_state=316)
    else:
        print("Unknown estimator")
        exit(0)


    if ml_procedure == "crossvalidate" or ml_procedure=="train" or \
                    ml_procedure=="trainandtest" or "featureimportance":
        [train_data, features, feature_names] = get_features(train_data)
        targets = get_targets(train_data, estimator)
    if ml_procedure == "test" or ml_procedure=="trainandtest":
        [test_data, test_features, none] = get_features(test_data)
        test_targets = get_targets(test_data, estimator)


    if ml_procedure == "crossvalidate":

        if cv_sampling == "cv_random":
            cv_split = ShuffleSplit(n_splits=5, test_size=0.25, random_state=121)
        elif cv_sampling == "cv_timeseries":
            split_arr = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1])
            train_split_f = lambda x: split_arr[train_data.index.day - 1]

            train_split = train_data.apply(train_split_f)["atemp"].values
            cv_split = PredefinedSplit(train_split)
            print(cv_split.unique_folds)
        else:
            print("Unknown cross validation fold setup.")
            exit(0)

        scores = cross_val_score(model, features, targets, cv=cv_split, scoring=RMSLE_scorer)

        if verbose==1:

            for train_index, test_index in cv_split.split():
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = features[train_index], features[test_index]
                y_train, y_test = targets[train_index], targets[test_index]
                model.fit(X_train, y_train)
                predicted = model.predict(X_test)

                y = y_test[:,0]
                fig, ax = plt.subplots()
                ax.scatter(y, predicted)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                plt.show()

                out_data = train_data.iloc[test_index.tolist()].copy(deep=True)
                out_data['prediction'] = pd.DataFrame(predicted, index=out_data.index, columns=['prediction'])
                out_data['ground_truth'] = pd.DataFrame(y, index=out_data.index, columns=['ground_truth'])
                out_data['diff'] = abs(out_data['ground_truth']-out_data['prediction'])
                out_data.to_csv("data/debug2.csv")

        print("Cross validation scores : ")
        print(scores)

    elif ml_procedure == "train":
        print("Not implemented.")
        pass#TODO

    elif ml_procedure == "test":
        print("Not implemented.")
        pass#TODO

    elif ml_procedure == "trainandtest":
        print("Training...")
        model.fit(features, targets)
        print("Testing...")
        predictions = model.predict(test_features)
        print("Saving results for submission...")
        test_data['count'] = predictions
        output_data = pd.DataFrame(test_data['count'], columns=['count'])
        output_data.to_csv(result_path, index=True, header=True)
    elif "featureimportance":
        model.fit(features, targets)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(features.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(features.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
        ord_feature_names = [(lambda x: feature_names[indices[x]])(x) for x in range(features.shape[1])]
        plt.xticks(range(features.shape[1]), ord_feature_names )
        plt.xlim([-1, features.shape[1]])
        plt.show()
    else:
        print("Unknown ml_procedure procedure.")
        exit(0)

