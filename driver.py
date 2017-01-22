# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:29:06 2017

@author: Matthew
"""

from argparse import ArgumentParser
import logging
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import sys


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
        if args.test_csv:
            train_data = pd.read_csv(args.train_csv, index_col='Id')
            test_data = pd.read_csv(args.test_data, index_col='Id')
        else:
            train_data = pd.read_csv(args.train_csv, index_col='Id')
            test_data = train_data.sample(frac=0.5)
            train_data = train_data.drop(test_data.index)
    except Exception as err:
        logging.exception(err)
        sys.exit(1)

    try:
        train_data = convert_to_floats(train_data)
        test_data = convert_to_floats(test_data)
        return train_data, test_data
    except Exception as err:
        logging.exception(err)
        sys.exit(1)


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
    data = data.fillna(-1000)
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


def gaussian_nb(house_train, price_train, house_test, price_answer):
    """ Runs GaussianNb model

        Input:
        house_train: Training data for house attributes
        price_train: Training data for prices
        house_test: Test data for house attributes
        price_answer: Test data for prices

        Output:
        A Dict with {'results': <price_gnb>, 'score': <score_gnb>}
    """
    gnb = GaussianNB()
    price_gnb = gnb.fit(house_train, price_train).predict(house_test)
    score_gnb = gnb.score(price_answer, price_gnb.reshape(-1, 1))
    print('Gaussian NB Score: {}'.format(score_gnb))
    return {'results': price_gnb, 'score': score_gnb}


def linear_regr(house_train, price_train, house_test, price_answer):
    """ Runs Linear Regression model

        Input:
        house_train: Training data for house attributes
        price_train: Training data for prices
        house_test: Test data for house attributes
        price_answer: Test data for prices

        Output:
        A Dict with {'results': <price_gnb>, 'score': <score_gnb>}
    """
    lrg = linear_model.LinearRegression()
    price_lrg = lrg.fit(house_train, price_train).predict(house_test)
    score_lrg = 0
    #score_lrg = lrg.score(price_answer, price_lrg.reshape(-1, 1))
    #print('Linear Regression Score: {}'.format(score_lrg))
    return {'results': price_lrg, 'score': score_lrg}


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
    for model in models:
        y_axis = (price_test - models[model]['results']).tolist()[0]
        x_axis = [x for x in range(len(y_axis))]
        plt.scatter(x_axis, y_axis)
        plt.title('{}: {}'.format(model, models[model]['score']))
        plt.tight_layout()
        if output_dir:
            plt.save(output_dir)
        else:
            plt.show()


def main(inargs=None):
    """ Main driver function """
    args = input_args(inargs)
    train_data, test_data = load_data(args)
    price_train, house_train = split_data(train_data)
    price_test, house_test = split_data(train_data)
    price_test = price_test.reshape(-1, 1)

    # Test models
    models = {}
    models['gnb'] = gaussian_nb(house_train, price_train,
                                house_test, price_test)
    models['linear_regr'] = linear_regr(house_train, price_train,
                                        house_test, price_test)

    # Compares models via plotting
    compare_data(price_test, models)

    return price_train, house_train, price_test, house_test, models


if __name__ == "__main__":
    price_train, house_train, price_test, house_test, models = main()
