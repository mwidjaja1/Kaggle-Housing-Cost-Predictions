#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:39:08 2017

@author: Matthew
"""

import logging
import math
import pandas as pd
import sys


def data(args):
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
