# Cargar Bibliotecas
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Cargar función
from src.features.featureEng import ingVariables

# Cargar datos
train_data_cln = pd.read_csv('./data/train_cln.csv')
test_data_cln = pd.read_csv('./data/test_cln.csv')

#Ingeniería de variables
train_data_ing = ingVariables(train_data_cln)
test_data_ing = ingVariables(test_data_cln)

#Juntar variables en train
train_data_ing['Condition_RRAn'] = train_data_ing['Condition1_RRAn'] + train_data_ing['Condition2_RRAn']
train_data_ing['Condition_RRAe'] = train_data_ing['Condition1_RRAe'] + train_data_ing['Condition2_RRAe']
train_data_ing['Condition_RRNn'] = train_data_ing['Condition1_RRNn'] + train_data_ing['Condition2_RRNn']
train_data_ing = train_data_ing.drop(['Condition1_RRAn', 'Condition2_RRAn', 'Condition1_RRAe', 
                                      'Condition2_RRAe', 'Condition1_RRNn', 'Condition2_RRNn'], axis=1)

#Agregar variables en test
test_data_ing['Condition_RRAn'] = test_data_ing['Condition1_RRAn'] 
test_data_ing['Condition_RRAe'] = test_data_ing['Condition1_RRAe'] 
test_data_ing['Condition_RRNn'] = test_data_ing['Condition1_RRNn'] 
test_data_ing['HouseStyle_2.5Fin'] = 0
test_data_ing['MiscFeature_TenC'] = 0
test_data_ing = test_data_ing.drop(['Condition1_RRAn', 'Condition1_RRAe', 'Condition1_RRNn'], axis=1)

# Ordenar columnas
columnas = train_data_ing.columns
test_data_ing = test_data_ing[columnas]

# Añadir variable objetivo
train_data_ing['SalePrice'] = train_data_cln['SalePrice']

# Guardar datos
train_data_ing.to_csv('./data/train_ing.csv', index=False)
test_data_ing.to_csv('./data/test_ing.csv', index=False)