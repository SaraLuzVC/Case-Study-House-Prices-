# Cargar Bibliotecas
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Cargar función
from src.data.data_cleaning import limpieza

# Cargar datos
train_data = pd.read_csv('./data/train_raw.csv')
test_data = pd.read_csv('./data/test_raw.csv')

# Limpieza de datos
train_data_cln = limpieza(train_data)
test_data_cln = limpieza(test_data)

# Agregar SalePrice
TdeC = 17.16 # 30-ene-2024
train_data_cln['SalePrice'] = train_data['SalePrice']
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TdeC,0)

# Guardar datos limpios
train_data_cln.to_csv('./data/train_cln.csv', index=False)
test_data_cln.to_csv('./data/test_cln.csv', index=False)

