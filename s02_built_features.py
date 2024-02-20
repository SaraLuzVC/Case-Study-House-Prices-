'''This script is used to build features from the cleaned data. It loads the
cleaned data, builds new features, and saves the new data in a new file.'''
# Cargar Bibliotecas
import warnings
import pandas as pd
import argparse
from src.utils import ing_variables, get_logger, no_file_error, save_file_error
warnings.filterwarnings("ignore")

# Configurar logging
logger = get_logger('built_features')
logger.info('Built features starting ...')

# Cargar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('train_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/train_cln.csv')
parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/test_cln.csv')
parser.add_argument('train_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/train_ing.csv')
parser.add_argument('test_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/test_ing.csv')
args = parser.parse_args()
#print(args.train_infile, args.test_infile, args.train_outfile, args.test_outfile)

# Cargar datos
logger.info(f"Cargando datos: {args.train_infile}")
train_data_cln = no_file_error(args.train_infile)
logger.info(f"Cargando datos: {args.test_infile}")
test_data_cln = no_file_error(args.test_infile)

# Ingeniería de variables
logger.info("Ingeniería de variables")
train_data_ing = ing_variables(train_data_cln)
test_data_ing = ing_variables(test_data_cln)

# Juntar variables en train
train_data_ing['Condition_RRAn'] = \
    train_data_ing['Condition1_RRAn'] + train_data_ing['Condition2_RRAn']
train_data_ing['Condition_RRAe'] = \
    train_data_ing['Condition1_RRAe'] + train_data_ing['Condition2_RRAe']
train_data_ing['Condition_RRNn'] = \
    train_data_ing['Condition1_RRNn'] + train_data_ing['Condition2_RRNn']
train_data_ing = train_data_ing.drop([
    'Condition1_RRAn', 'Condition2_RRAn',
    'Condition1_RRAe', 'Condition2_RRAe',
    'Condition1_RRNn', 'Condition2_RRNn'], axis=1)

# Agregar variables en test
test_data_ing['Condition_RRAn'] = test_data_ing['Condition1_RRAn']
test_data_ing['Condition_RRAe'] = test_data_ing['Condition1_RRAe']
test_data_ing['Condition_RRNn'] = test_data_ing['Condition1_RRNn']
test_data_ing['HouseStyle_2.5Fin'] = 0
test_data_ing['MiscFeature_TenC'] = 0
test_data_ing = test_data_ing.drop([
    'Condition1_RRAn', 'Condition1_RRAe',
    'Condition1_RRNn'], axis=1)

# Ordenar columnas
columnas = train_data_ing.columns
test_data_ing = test_data_ing[columnas]

# Añadir variable objetivo
train_data_ing['SalePrice'] = train_data_cln['SalePrice']

# Guardar datos
logger.info(f"Guardando datos: {args.train_outfile}")
save_file_error(args.train_outfile, train_data_ing)
logger.info(f"Guardando datos: {args.test_outfile}")
save_file_error(args.test_outfile, test_data_ing)
