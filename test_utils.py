'''This script is used to test the functions in the utils.py file'''
import warnings
# import sys
# import os
# import logging
from os.path import exists
import pandas as pd
import numpy as np
# Add the path to the src folder to the sys.path list
# sys.path.append(os.path.abspath
# (os.path.join(os.path.dirname(__file__), '..')))
from src.utils import num_vars, limpieza, ing_variables, get_logger
warnings.filterwarnings("ignore")

# logging.basicConfig(
#     filename="../test.log",
#     level=logging.DEBUG,
#     filemode='w',
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
# logger = logging.getLogger(__name__)
# handler = logging.FileHandler("./logs/test.log")

logger = get_logger('test')

TRAIN = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6],
                      'SalePrice': [7, 8, 9]})
TEST = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
# file_trn = './data/train.csv'
# file_tst = './data/test.csv'
# logger =get_logger('test')

data_trn = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')
clean = limpieza(data_trn)
clean_test = limpieza(data_test)
ing_var = ing_variables(clean)


def test_cols_limpieza():
    '''This function tests if the columns in the clean train dataset
    are the same as the columns in the clean_test dataset.'''
    assert np.all(clean.columns == clean_test.columns)


def test_cols_entrada_train():
    '''This function tests if the columns in the train dataset
    are the ones expected.'''
    assert np.all(data_trn.columns == [
       'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'])


def test_cols_entrada_test():
    '''This function tests if the columns in the test dataset
    are the ones expected.'''
    assert np.all(data_test.columns == [
       'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition'])


def test_train_path_file():
    '''This function tests if the train.csv file exists.'''
    assert exists('../data/train.csv') is True


def test_test_path_file():
    '''This function tests if the test.csv file exists.'''
    assert exists('../data/test.csv') is True


# test no_file_error
def test_no_file_error():
    '''This function tests the no_file_error function in the
    utils.py file. It checks if the function returns a DataFrame object.'''
    assert isinstance(data_trn, pd.DataFrame)

# # test limpieza
# def test_limpieza():
#     '''This function tests the limpieza function in the
#     utils.py file. It checks if the function returns a DataFrame object.'''
#     assert type(clean)==pd.DataFrame

# # test ing_variables
# def test_ing_variables():
#     '''This function tests the ing_variables function in the
#     utils.py file. It checks if the function returns a DataFrame object.'''
#     assert type(ing_var)==pd.DataFrame


# test num_vars
def test_num_vars():
    '''This function tests the num_vars function in the
    utils.py file. It checks if the function returns a boolean value
    when comparing the number of variables in the
    training and test datasets.'''
    assert num_vars(data_trn, data_test, logger) is True


# test num_obs
def test_num_obs():
    '''This function tests the number of rows in the training dataset.'''
    assert data_trn.shape[0] > 0


# test num_cols_2:
def test_num_cols_2():
    '''This function tests the number of columns in the training dataset.'''
    assert data_trn.shape[1] == 81


def test_num_obs_2():
    '''This function tests the number of rows in the test dataset.'''
    assert data_test.shape[0] > 0


# test num_cols_2:
def test_num_cols_3():
    '''This function tests the number of columns in the test dataset.'''
    assert data_test.shape[1] == 80
