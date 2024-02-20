import pytest
import warnings
import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import num_obs, num_vars, no_file_error,  get_logger, limpieza, ing_variables
warnings.filterwarnings("ignore")

##############################
# test get_logger
def test_get_logger():
    logger = get_logger('prep')
    assert(type(logger)==logging.Logger)



data_tra = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'SalePrice': [7, 8, 9]})
data_tst = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

file_trn = './data/train.csv'
file_tst = './data/test.csv'
logger = get_logger('test')

data_trn = no_file_error(file_trn, logger)
clean = limpieza(data_trn)
ing_var = ing_variables(clean)

# test no_file_error
def test_no_file_error():
    assert(type(data_trn)==pd.DataFrame)

# test limpieza
def test_limpieza():
    assert(type(clean)==pd.DataFrame)
    
# test ing_variables
def test_ing_variables():
    assert(type(ing_var)==pd.DataFrame)
    
# test num_vars
def test_num_vars():
    assert num_vars(data_tra, data_tst, logger)==True

# test num_obs
def test_num_obs():
    assert num_obs(data_tra, file_tst, logger)==True
