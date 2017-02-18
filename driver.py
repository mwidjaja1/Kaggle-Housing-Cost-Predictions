# -*- coding: utf-8 -*-
"""
--test_csv /Users/Matthew/Github/Kaggle_Housing/test.csv
"""

import datetime as dt
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd

import data
import inputargs
import model


def compare_data(price_test, models, output_dir=None):
    """ Plots the difference between each price in the test set versus model

        Inputs:
        price_test: Actual test data for prices
        models: Dict of model results in the format
                {'results': <price_gnb>, 'score': <score_gnb>}
        title: Title of plot

        Output:
        A plot saved to output_dir if specified
    """
    for key in models:
        plt.scatter(price_test, models[key]['results'])
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title('{}: {}'.format(model, models[key]['score']))
        plt.tight_layout()
        if output_dir:
            plt.save(output_dir)
        else:
            plt.show()


def output_data(house_test, price_model):
    """ Exports the prices estimated by the model """
    date = dt.datetime.now().strftime('%m%d_%H%M')
    out_df = pd.DataFrame(price_model['results'], index=house_test.index,
                          columns=['SalePrice'])
    out_df.to_csv('data_{}.csv'.format(date))


def main(args):
    """ Main driver function """
    all_data, train_data, test_data = data.load(args)
    price_train, house_train = data.split(train_data)
    price_test, house_test = data.split(test_data)

    features = model.feature_tree(house_train, price_train)
    house_train = house_train[features]
    house_test = house_test[features]

    # Test models
    models = {}
    max_score = (None, {}, 0.0)
    # opt = (1500, 15)
    for opt in itertools.product([1500], list(range(13, 20))):
        models['rf_regr'] = model.ada_boost_regr(house_train, price_train,
                                                 house_test, price_test, opt)
        if models['rf_regr']['score'] > max_score[2]:
            print(opt)
            max_score = (opt, models, models['rf_regr']['score'])

    # Compares models via plotting and/or output results
    if not args.test_csv:
        compare_data(price_test, models)
        # print('Best Results were {} with {}'.format(max_score[2], max_score[0]))
        # compare_data(price_test, max_score[1])
    else:
        output_data(house_test, models['rf_regr'])

    return all_data, price_train, house_train, price_test, house_test, models


if __name__ == "__main__":
    args = inputargs.parse()
    all_data, price_train, house_train, price_test, house_test, models = \
        main(args)
