''' This code is used to clean the data and save it in a new file. '''

# Cargar Bibliotecas
import warnings
import pandas as pd
from src.utils import limpieza
warnings.filterwarnings("ignore")

# Cargar datos
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Limpieza de datos
train_data_cln = limpieza(train_data)
test_data_cln = limpieza(test_data)

# Agregar SalePrice
TIPO_DE_CAMBIO = 17.16  # 30-ene-2024
train_data_cln['SalePrice'] = train_data['SalePrice']
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TIPO_DE_CAMBIO, 0)

# Guardar datos limpios
train_data_cln.to_csv('./data/train_cln.csv', index=False)
test_data_cln.to_csv('./data/test_cln.csv', index=False)
