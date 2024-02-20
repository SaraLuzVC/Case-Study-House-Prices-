''' This code is used to clean the data and save it in a new file. '''

# Cargar Bibliotecas
import warnings
import pandas as pd
import argparse
from src.utils import limpieza
warnings.filterwarnings("ignore")

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
train_data = pd.read_csv(args.train_infile)
test_data = pd.read_csv(args.test_infile)

# Limpieza de datos
train_data_cln = limpieza(train_data)
test_data_cln = limpieza(test_data)

# Agregar SalePrice
TIPO_DE_CAMBIO = 17.16  # 30-ene-2024
train_data_cln['SalePrice'] = train_data['SalePrice']
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TIPO_DE_CAMBIO, 0)

# Guardar datos limpios
train_data_cln.to_csv(args.train_outfile, index=False)
test_data_cln.to_csv(args.test_outfile, index=False)
