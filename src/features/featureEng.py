# Cargar Bibliotecas
import pandas as pd
import numpy as np

def ingVariables(data_cln):
        # Seleccionar variables de interés
        data_ing = data_cln[['LotFrontage', 'LotArea', 'Street', 'Utilities', 
        'YearBuilt', 'BedroomAbvGr',
        'KitchenQual', 'GarageCars', 'PavedDrive', 
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'MiscFeature', 'SaleCondition']]

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
        data_ing['OverallQual'] = data_cln['OverallQual'].map(dic_qual)
        data_ing['OverallCond'] = data_cln['OverallCond'].map(dic_qual)

        # Con o sin sótano
        data_ing['sotano'] =  np.where(data_cln['BsmtCond']=='NA', 0, 1)

        # Creamos las variables baños completos, baños medios, m2 construidos y m2 de terraza
        data_ing['banos_completos'] = data_cln['BsmtFullBath'] + data_cln['FullBath']
        data_ing['banos_medios'] = data_cln['BsmtHalfBath'] + data_cln['HalfBath']
        data_ing['m2construidos'] = data_cln['TotalBsmtSF'] + data_cln['GrLivArea']
        data_ing['m2terraza'] = data_cln['WoodDeckSF'] + data_cln['OpenPorchSF'] + data_cln['EnclosedPorch'] + data_cln['3SsnPorch'] + data_cln['ScreenPorch']

        # Convertimos a ordinal algunas variables
        dic_kitchen ={'Ex' : 5, 'Gd' : 4, 'TA': 3, 'Fa': 2, 'Po' : 1}
        data_ing['KitchenQual'] = data_cln['KitchenQual'].map(dic_kitchen)
        dic_util ={'AllPub' : 4, 'NoSewr' : 3, 'NoSeWa': 2, 'ELO': 1}
        data_ing['Utilities'] = data_cln['Utilities'].map(dic_util)
        dic_street ={'Grvl' : 0, 'Pave' : 1}
        data_ing['Street'] = data_cln['Street'].map(dic_street)
        dic_drive ={'Y' : 1, 'P' : 2, 'N': 3}
        data_ing['PavedDrive'] = data_cln['PavedDrive'].map(dic_drive)

        # Convertimos a one hot encoding las variables categóricas
        data_ing = pd.get_dummies(data_ing, columns = ['Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'MiscFeature', 'SaleCondition']) 

        data_ing['Condition_Norm'] = data_ing['Condition1_Norm'] + data_ing['Condition2_Norm']
        data_ing['Condition_Feedr'] = data_ing['Condition1_Feedr'] + data_ing['Condition2_Feedr']
        data_ing['Condition_Artery'] = data_ing['Condition1_Artery'] + data_ing['Condition2_Artery']
        data_ing['Condition_PosN'] = data_ing['Condition1_PosN'] + data_ing['Condition2_PosN']
        data_ing['Condition_PosA'] = data_ing['Condition1_PosA'] + data_ing['Condition2_PosA']

        data_ing = data_ing.drop(['Condition1_Norm', 'Condition2_Norm', 'Condition1_Feedr', 'Condition2_Feedr', 
                                  'Condition1_Artery', 'Condition2_Artery',  
                                  'Condition1_PosN', 'Condition2_PosN', 'Condition1_PosA', 
                                  'Condition2_PosA'], axis=1)
        return data_ing
