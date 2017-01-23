# -*- coding: utf-8 -*-
"""
--test_csv /Users/Matthew/Github/Kaggle_Housing/test.csv
"""

from argparse import ArgumentParser
import datetime as dt
import itertools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
import sys

import model


def input_args(inargs):
    """ Input arguments parser """
    parser = ArgumentParser("./driver.py")
    parser.add_argument('train_csv', help='Path to the CSV training data file')
    parser.add_argument('--test_csv', default=None,
                        help='Path to the CSV test data file.')

    if inargs:
        args = parser.parse_args(inargs)
    else:
        args = parser.parse_args()
    return args


def load_data(args):
    """ Loads CSV data via Pandas and converts all values to floats

        Input:
        args: Input arguments for script. Uses train_csv and test_csv

        Output:
        train_data: DataFrame with the training data
        test_data: DataFrame with the test data. If --test_csv is not set,
                   this will be 50% of the data & train_data is the other 50%.
    """
    try:
        all_data = pd.read_csv(args.train_csv, index_col=0)
    except Exception as err:
        logging.exception(err)
        sys.exit(1)

    try:
        if args.test_csv:
            test_data = pd.read_csv(args.test_csv, index_col=0)
            train_data = remove_outliers(all_data)
        else:
            test_data = all_data.sample(frac=0.4)
            train_data = remove_outliers(all_data.drop(test_data.index))
    except Exception as err:
        logging.exception(err)
        sys.exit(1)

    try:
        train_data = convert_to_floats(train_data)
        test_data = convert_to_floats(test_data)
        return all_data, train_data, test_data
    except Exception as err:
        logging.exception(err)
        sys.exit(1)


def remove_outliers(data):
    """ Removes the top & bottom 5% of sale prices from the data

        Input/Output:
        data: DataFrame to clean data from, must have 'SalePrice' column
    """
    try:
        perc = np.percentile(data['SalePrice'], [5, 95])
        cleaned_data = data[(data['SalePrice'] > perc[0]) &
                            (data['SalePrice'] < perc[1])]
    except Exception as err:
        logging.exception(err)
        sys.exit(1)
    return cleaned_data


def convert_to_floats(data):
    """ Converts all values, be they strings or NaN values to floats

        Input/Output:
        data: DataFrame to convert
    """
    convert = {}

    for col in data.columns:
        # Finds first nan value in this given column
        for idx in data.index:
            value = data.loc[idx, col]
            if (not isinstance(value, float) or
               (isinstance(value, float) and not math.isnan(value))):
                break

        # Checks if this is a numerical type or a string.
        try:
            if isinstance(data[col][idx], str):
                print('Will convert {} to ints'.format(col))
                unique_vals = data[col].unique()
                convert[col] = {x: i for i, x in enumerate(unique_vals)}
        except Exception as err:
            logging.exception(err)
            sys.exit(1)

    # Converts all strings to integers and fills nan values
    data = data.replace(convert)
    data = data.fillna(0)
    return data


def split_data(data):
    """ Splits a DataFrame into two DataFrames for house attributes & prices.

        Input:
        data: A DataFrame with the column 'SalePrice' which will be split out.
              If the DataFrame doesn't have this column, we won't split it.

        Output:
        price_data: DataFrame with the 'SalePrice' column if it exists.
                    Otherwise, this will be None.
        house_data: DataFrame with the house attributes

    """
    try:
        if 'SalePrice' in data.columns:
            price_data = data['SalePrice']
            house_data = data.loc[:, data.columns != 'SalePrice']
        else:
            house_data = data
            price_data = None
        return price_data, house_data
    except Exception as err:
        logging.exception(err)
        sys.exit(1)


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


def main(inargs=None):
    """ Main driver function """
    args = input_args(inargs)
    all_data, train_data, test_data = load_data(args)
    price_train, house_train = split_data(train_data)
    price_test, house_test = split_data(test_data)

    # Test models
    models = {}
    # max_score = (None, {}, 0.0)
    # for opt in itertools.product(list(range(1, 21)), list(range(1, 21))):
    opt = (12, 18)
    models['rf_class'] = model.random_forest(house_train, price_train,
                                             house_test, price_test, opt)
    models['rf_regr'] = model.random_forest_regr(house_train, price_train,
                                                 house_test, price_test, opt)
    # if models['rf_regr']['score'] > max_score[2]:
    #    max_score = (opt, models, models['rf_regr']['score'])

    # Compares models via plotting and/or output results
    print
    if not args.test_csv:
        compare_data(price_test, models)
        # print('Best Results were {} with {}'.format(max_score[2], max_score[0]))
        # compare_data(price_test, max_score[1])
    else:
        output_data(house_test, models['tree'])

    return all_data, price_train, house_train, price_test, house_test, models


if __name__ == "__main__":
    all_data, price_train, house_train, price_test, house_test, models = main()
