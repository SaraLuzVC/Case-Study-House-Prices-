# Cargar Bibliotecas
import pandas as pd
import numpy as np

# Limpieza de datos
def limpieza(datos):
    
    # Eliminar columnas que no se utilizarán
    data_cln = datos[['LotFrontage', 'LotArea', 'Street',
            'Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
        'BsmtCond',  'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr',  'KitchenQual',
        'GarageCars', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscFeature', 
        'SaleCondition']]

    # Conversión de unidades
    # Equivalencias
    ft2 = 0.092903 # metros cuadrados
    ft = 0.3048 # metros
    # Conversión
    data_cln['LotFrontage'] = datos['LotFrontage']*ft
    data_cln['LotArea'] = datos['LotArea']*ft2
    data_cln['TotalBsmtSF'] = datos['TotalBsmtSF']*ft2
    data_cln['GrLivArea'] = datos['GrLivArea']*ft2
    data_cln['WoodDeckSF'] = datos['WoodDeckSF']*ft2
    data_cln['OpenPorchSF'] = datos['OpenPorchSF']*ft2
    data_cln['EnclosedPorch'] = datos['EnclosedPorch']*ft2
    data_cln['3SsnPorch'] = datos['3SsnPorch']*ft2
    data_cln['ScreenPorch'] = datos['ScreenPorch']*ft2

    # Imputación de datos faltantes
    # Rellenamos colocando la opción 'NA'
    data_cln['BsmtCond'] = data_cln['BsmtCond'].fillna('NA')
    data_cln['MiscFeature'] = data_cln['MiscFeature'].fillna('NA')
    # Rellenamos con el promedio
    for c in data_cln.columns:
        if(data_cln[c].dtype!='object'):
            data_cln[c].fillna(data_cln[c].mean(), inplace = True)
    return data_cln