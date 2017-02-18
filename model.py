# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:57:21 2017

@author: Matthew
"""

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, ensemble, tree


def feature_tree(house_train, price_train):
    """ Uses a Extra Trees Classifier to determine which feature are most
        helpful
    """
    etc = ensemble.ExtraTreesClassifier()
    etc = etc.fit(house_train, price_train)
    feats = pd.DataFrame(etc.feature_importances_, index=house_train.columns)
    feats = feats.sort_values(0, ascending=0)
    return feats[feats[0] > 0.01].index


def random_forest(house_train, price_train, house_test, price_test,
                  opt=(10, None)):
    """ Runs the Random Forest Classifier model

        Input:
        house_train: Training data for house attributes
        price_train: Training data for prices
        house_test: Test data for house attributes
        price_test: Test data for prices
        opt: (#NEstimator, #MaxDepth). Default will use (10, None)

        Output:
        A Dict with {'results': <price_gnb>, 'score': <score_gnb>}
    """
    clf = ensemble.RandomForestClassifier(n_estimators=opt[0],
                                          max_depth=opt[1])
    clf = clf.fit(house_train, price_train)
    price_clf = clf.predict(house_test)

    if isinstance(price_test, np.ndarray):
        score_clf = metrics.accuracy_score(price_clf, price_test)
        print('Random Forest Classifier for ({}): {}'.format(opt, score_clf))
    else:
        score_clf = 0
    return {'results': price_clf, 'score': score_clf}


def random_forest_regr(house_train, price_train, house_test, price_test,
                       opt=(10, None)):
    """ Runs the Random Forest Regression model

        Input:
        house_train: Training data for house attributes
        price_train: Training data for prices
        house_test: Test data for house attributes
        price_test: Test data for prices
        opt: (#NEstimator, #MaxDepth). Default will use (10, None)

        Output:
        A Dict with {'results': <price_gnb>, 'score': <score_gnb>}
    """
    clf = ensemble.RandomForestRegressor(n_estimators=opt[0], max_depth=opt[1])
    clf = clf.fit(house_train, price_train)
    price_clf = clf.predict(house_test)

    if isinstance(price_test, np.ndarray):
        score_clf = metrics.r2_score(price_clf, price_test)
        print('Random Forest Regression for {}: {}'.format(opt, score_clf))
    else:
        score_clf = 0
    return {'results': price_clf, 'score': score_clf}


def ada_boost_regr(house_train, price_train, house_test, price_test,
                   opt=(10, None)):
    """ Runs the Ada Boost Random Forest Regression model

        Input:
        house_train: Training data for house attributes
        price_train: Training data for prices
        house_test: Test data for house attributes
        price_test: Test data for prices
        opt: (#NEstimator, #MaxDepth). Default will use (10, None)

        Output:
        A Dict with {'results': <price_gnb>, 'score': <score_gnb>}
    """
    clf = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=opt[1]),
                                     n_estimators=opt[0])
    scores = model_selection.cross_val_score(clf, house_train, price_train)
    price_clf = clf.fit(house_train, price_train).predict(house_test)

    try:
        score_clf = scores.mean()
    except Exception:
        score_clf = 0
    print('Random Forest Regression for {}: {}'.format(opt, score_clf))
    return {'results': price_clf, 'score': score_clf}
