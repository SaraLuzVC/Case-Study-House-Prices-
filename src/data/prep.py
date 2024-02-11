# Cargar Bibliotecas
import pandas as pd
import numpy as np

# Cargar datos
train_data = pd.read_csv('../../data/train_raw.csv')

# Limpieza de datos
# Eliminar columnas que no se utilizarán
train_data_cln = train_data[['LotFrontage', 'LotArea', 'Street',
        'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
       'BsmtCond',  'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr',  'KitchenQual',
       'GarageCars', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscFeature', 
       'SaleCondition', 'SalePrice']]

# Conversión de unidades
# Equivalencias
TdeC = 17.16 # 30-ene-2024
ft2 = 0.092903 # metros cuadrados
ft = 0.3048 # metros
# Conversión
train_data_cln['LotFrontage'] = train_data['LotFrontage']*ft
train_data_cln['LotArea'] = train_data['LotArea']*ft2
train_data_cln['TotalBsmtSF'] = train_data['TotalBsmtSF']*ft2
train_data_cln['GrLivArea'] = train_data['GrLivArea']*ft2
train_data_cln['WoodDeckSF'] = train_data['WoodDeckSF']*ft2
train_data_cln['OpenPorchSF'] = train_data['OpenPorchSF']*ft2
train_data_cln['EnclosedPorch'] = train_data['EnclosedPorch']*ft2
train_data_cln['3SsnPorch'] = train_data['3SsnPorch']*ft2
train_data_cln['ScreenPorch'] = train_data['ScreenPorch']*ft2
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TdeC,0)

# Imputación de datos faltantes
# Rellenamos colocando la opción 'NA'
train_data_cln['BsmtCond'] = train_data_cln['BsmtCond'].fillna('NA')
train_data_cln['MiscFeature'] = train_data_cln['MiscFeature'].fillna('NA')
# Rellenamos con el promedio
for c in train_data_cln.columns:
    if(train_data_cln[c].dtype!='object'):
        train_data_cln[c].fillna(train_data_cln[c].mean(), inplace = True)
        
# Guardar datos limpios
train_data_cln.to_csv('../../data/train_cln.csv', index=False)
