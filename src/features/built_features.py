# Cargar Bibliotecas
import pandas as pd
import numpy as np

# Cargar datos
train_data_cln = pd.read_csv('../../data/train_cln.csv')

# Seleccionar variables de interés
train_data_ing = train_data_cln[['LotFrontage', 'LotArea', 'Street', 'Utilities', 
       'YearBuilt', 'BedroomAbvGr',
       'KitchenQual', 'GarageCars', 'PavedDrive', 
       'SalePrice', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'MiscFeature', 'SaleCondition']]

# Reducir opciones
dic_qual ={1 : 3, 
        2 : 3,
        3 : 3,
        4 : 3,
        5 : 2,
        6 : 2,
        7 : 2,
        8 : 1,
        9 : 1,
        10 : 1}
train_data_ing['OverallQual'] = train_data_cln['OverallQual'].map(dic_qual)
train_data_ing['OverallCond'] = train_data_cln['OverallCond'].map(dic_qual)

# Con o sin sótano
train_data_ing['sotano'] =  np.where(train_data_cln['BsmtCond']=='NA', 0, 1)

# Creamos las variables baños completos, baños medios, m2 construidos y m2 de terraza
train_data_ing['banos_completos'] = train_data_cln['BsmtFullBath'] + train_data_cln['FullBath']
train_data_ing['banos_medios'] = train_data_cln['BsmtHalfBath'] + train_data_cln['HalfBath']
train_data_ing['m2construidos'] = train_data_cln['TotalBsmtSF'] + train_data_cln['GrLivArea']
train_data_ing['m2terraza'] = train_data_cln['WoodDeckSF'] + train_data_cln['OpenPorchSF'] + train_data_cln['EnclosedPorch'] + train_data_cln['3SsnPorch'] + train_data_cln['ScreenPorch']

# Convertimos a ordinal algunas variables
dic_kitchen ={'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa': 2, 'Po' : 1}
train_data_ing['KitchenQual'] = train_data_cln['KitchenQual'].map(dic_kitchen)
dic_util ={'AllPub' : 4, 'NoSewr' : 3, 'NoSeWa': 2, 'ELO': 1}
train_data_ing['Utilities'] = train_data_cln['Utilities'].map(dic_util)
dic_street ={'Grvl' : 0, 'Pave' : 1}
train_data_ing['Street'] = train_data_cln['Street'].map(dic_street)
dic_drive ={'Y' : 1, 'P' : 2, 'N': 3}
train_data_ing['PavedDrive'] = train_data_cln['PavedDrive'].map(dic_drive)

# Convertimos a one hot encoding las variables categóricas
train_data_ing = pd.get_dummies(train_data_ing, columns = ['Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'MiscFeature', 'SaleCondition']) 

train_data_ing['Condition_Norm'] = train_data_ing['Condition1_Norm'] + train_data_ing['Condition2_Norm']
train_data_ing['Condition_Feedr'] = train_data_ing['Condition1_Feedr'] + train_data_ing['Condition2_Feedr']
train_data_ing['Condition_Artery'] = train_data_ing['Condition1_Artery'] + train_data_ing['Condition2_Artery']
train_data_ing['Condition_RRAn'] = train_data_ing['Condition1_RRAn'] + train_data_ing['Condition2_RRAn']
train_data_ing['Condition_PosN'] = train_data_ing['Condition1_PosN'] + train_data_ing['Condition2_PosN']
train_data_ing['Condition_RRAe'] = train_data_ing['Condition1_RRAe'] + train_data_ing['Condition2_RRAe']
train_data_ing['Condition_PosA'] = train_data_ing['Condition1_PosA'] + train_data_ing['Condition2_PosA']
train_data_ing['Condition_RRNn'] = train_data_ing['Condition1_RRNn'] + train_data_ing['Condition2_RRNn']

train_data_ing = train_data_ing.drop(['Condition1_Norm', 'Condition2_Norm', 'Condition1_Feedr', 'Condition2_Feedr', 'Condition1_Artery', 'Condition2_Artery', 'Condition1_RRAn', 'Condition2_RRAn', 'Condition1_PosN', 'Condition2_PosN', 'Condition1_RRAe', 'Condition2_RRAe', 'Condition1_PosA', 'Condition2_PosA', 'Condition1_RRNn', 'Condition2_RRNn'], axis=1)
train_data_ing.columns

# Guardar datos
train_data_ing.to_csv('../../data/train_ing.csv', index=False)
