''' This code is used to clean the data and save it in a new file. '''

# Cargar Bibliotecas
import warnings
import pandas as pd
import argparse
import logging
from datetime import datetime
from src.utils import limpieza, get_logger, no_file_error, save_file_error
warnings.filterwarnings("ignore")

# Configurar logging
logger = get_logger('prep')
logger.info('Prep starting ...')
    
    
    # Read inputs
logger.info("Prep starting ...")
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
x=2
logger.debug(f"The value of x is {x}")

try:
    1/0
except ZeroDivisionError as e:
    logger.error(f"Error: {e}")

# Cargar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('train_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/train.csv')
parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/test.csv')
parser.add_argument('train_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/train_cln.csv')
parser.add_argument('test_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/test_cln.csv')
args = parser.parse_args()
#print(args.train_infile, args.test_infile, args.train_outfile, args.test_outfile)

# Cargar datos
logger.info(f"Cargando datos: {args.train_infile}")
train_data = no_file_error(args.train_infile)
logger.info(f"Cargando datos: {args.test_infile}")
test_data = no_file_error(args.test_infile)

# Limpieza de datos
logger.info("Limpieza de datos")
train_data_cln = limpieza(train_data)
test_data_cln = limpieza(test_data)

# Agregar SalePrice
TIPO_DE_CAMBIO = 17.16  # 30-ene-2024
logger.info(f"El tipo de cambio es {TIPO_DE_CAMBIO}")
train_data_cln['SalePrice'] = train_data['SalePrice']
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TIPO_DE_CAMBIO, 0)

# Guardar datos limpios
logger.info(f"Guardando datos: {args.train_outfile}")
save_file_error(args.train_outfile, train_data_cln)
logger.info(f"Guardando datos: {args.test_outfile}")
save_file_error(args.test_outfile, test_data_cln)
