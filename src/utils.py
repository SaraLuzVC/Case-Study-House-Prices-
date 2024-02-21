'''This script has the functions to clean the data and to select
the variables of interest.

The functions are:
    - num_vars: This function checks if the number of variables in the training
    data is one more than the number of variables in the test data.
    - num_obs: This function checks if the number of observations in the data
    is greater than 0.
    - no_file_error: This function checks if the file exists.
    - save_file_error: This function checks if the file was saved.
    - get_logger: This function configures the logging module to save the logs
    in a file.
    - get_best_score: This function gets the best score and related information
    from a GridSearchCV object.
    - ing_variables: This function selects the variables of interest and
    transforms them to the format that will be used to train the models.
    - limpieza: This function cleans the data. It selects the variables
    of interest and transforms them to the format that will be used to
    train the models.'''

# Cargar Bibliotecas
import logging
from datetime import datetime
import pandas as pd
import numpy as np


# Numero de variables
def num_vars(df_train, df_test, logger):
    '''This function checks if the number of variables in the training data is
    one more than the number of variables in the test data. If the number of
    variables in the training data is one more than the number of variables
    in the test data, the function returns True. If the number of variables in
    the training data is not one more than the number of variables in the test
    data, the function raises a ValueError and returns False.'''
    if df_train.shape[1] == df_test.shape[1]+1:
        logger.info(f"Numero de variables en train: {df_train.shape[1]}")
        logger.info(f"Numero de variables en test: {df_test.shape[1]}")
        return True
    logger.error(f"Numero de variables en train: {df_train.shape[1]}")
    logger.error(f"Numero de variables en test: {df_test.shape[1]}")
    raise ValueError("Numero de variables en train y test no coincide")


# Numero de observaciones
def num_obs(data, file, logger):
    '''This function checks if the number of observations in the data is
    greater than 0. If the number of observations in the data is greater
    than 0, the function returns True. If the number of observations in
    the data is not greater than 0, the function raises a ZeroDivisionError
    and returns False.'''
    try:
        logger.info(f"Numero de observaciones en {file}: {data.shape[0]}")
        return True
    except ZeroDivisionError as exc:
        raise ZeroDivisionError(
            f"Numero de observaciones en {file} es 0 {exc}"
            ) from exc


# Función para manejar errores de archivos
def no_file_error(file, logger):
    '''This function checks if the file exists. If the file exists, the
    function returns True. If the file does not exist, the function raises
    a FileNotFoundError and returns False.'''
    try:
        data = pd.read_csv(file)
    except FileNotFoundError as exc:
        logger.error("Error: %s no existe, %s", file, exc)
        raise FileNotFoundError(
            f"Error: {file} no existe, {exc}"
            ) from exc
    return data


# Función para manejar errores de archivos
def save_file_error(file, data, logger):
    '''This function checks if the file was saved. If the file was
    saved, the function returns True. If the file was not saved, the
    function returns False.'''
    try:
        data.to_csv(file, index=False)
        return True
    except FileNotFoundError:
        logger.error(f"No se pudo guardar el archivo {file}")
        return False


# Función para configurar logging
def get_logger(archivo_log):
    '''This function configures the logging module to save
    the logs in a file'''
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    log_train_file_name = f"logs/{date_time}_{archivo_log}.log"
    logging.basicConfig(
        filename=log_train_file_name,
        level=logging.DEBUG,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(log_train_file_name)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Función para obtener el mejor score
def get_best_score(grid):
    """
    Get the best score and related information from a GridSearchCV object.

    Args:
        grid (GridSearchCV): The GridSearchCV object.

    Returns:
        float: Best score.
    """
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)
    print(grid.best_params_)
    print(grid.best_estimator_)
    return best_score


def ing_variables(data_cln):
    '''
    This function selects the variables of interest and transforms them to
    the format that will be used to train the models.

    Args:
        data_cln (pd.DataFrame): The cleaned data.

    Returns:
        pd.DataFrame: The data with the selected variables.
    '''
    # Seleccionar variables de interés
    data_ing = data_cln[['LotFrontage', 'LotArea', 'Street', 'Utilities',
                         'YearBuilt', 'BedroomAbvGr', 'KitchenQual',
                         'GarageCars', 'PavedDrive', 'Neighborhood',
                         'Condition1', 'Condition2', 'BldgType',
                         'HouseStyle', 'MiscFeature', 'SaleCondition']]

    # Reducir opciones
    dic_qual = {1: 3,
                2: 3,
                3: 3,
                4: 3,
                5: 2,
                6: 2,
                7: 2,
                8: 1,
                9: 1,
                10: 1}
    data_ing['OverallQual'] = data_cln['OverallQual'].map(dic_qual)
    data_ing['OverallCond'] = data_cln['OverallCond'].map(dic_qual)

    # Con o sin sótano
    data_ing['sotano'] = np.where(data_cln['BsmtCond'] == 'NA', 0, 1)

    # Creamos las variables baños completos, baños medios,
    # m2 construidos y m2 de terraza
    data_ing['banos_completos'] = \
        data_cln['BsmtFullBath'] + data_cln['FullBath']
    data_ing['banos_medios'] = \
        data_cln['BsmtHalfBath'] + data_cln['HalfBath']
    data_ing['m2construidos'] = \
        data_cln['TotalBsmtSF'] + data_cln['GrLivArea']
    data_ing['m2terraza'] = \
        data_cln['WoodDeckSF'] + data_cln['OpenPorchSF'] + \
        data_cln['EnclosedPorch'] + data_cln['3SsnPorch'] + \
        data_cln['ScreenPorch']

    # Convertimos a ordinal algunas variables
    dic_kitchen = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    data_ing['KitchenQual'] = data_cln['KitchenQual'].map(dic_kitchen)
    dic_util = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
    data_ing['Utilities'] = data_cln['Utilities'].map(dic_util)
    dic_street = {'Grvl': 0, 'Pave': 1}
    data_ing['Street'] = data_cln['Street'].map(dic_street)
    dic_drive = {'Y': 1, 'P': 2, 'N': 3}
    data_ing['PavedDrive'] = data_cln['PavedDrive'].map(dic_drive)

    # Convertimos a one hot encoding las variables categóricas
    data_ing = pd.get_dummies(data_ing, columns=[
            'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
            'HouseStyle', 'MiscFeature', 'SaleCondition'])

    data_ing['Condition_Norm'] = \
        data_ing['Condition1_Norm'] + data_ing['Condition2_Norm']
    data_ing['Condition_Feedr'] = \
        data_ing['Condition1_Feedr'] + data_ing['Condition2_Feedr']
    data_ing['Condition_Artery'] = \
        data_ing['Condition1_Artery'] + data_ing['Condition2_Artery']
    data_ing['Condition_PosN'] = \
        data_ing['Condition1_PosN'] + data_ing['Condition2_PosN']
    data_ing['Condition_PosA'] = \
        data_ing['Condition1_PosA'] + data_ing['Condition2_PosA']

    data_ing = data_ing.drop([
        'Condition1_Norm', 'Condition2_Norm', 'Condition1_Feedr',
        'Condition2_Feedr', 'Condition1_Artery', 'Condition2_Artery',
        'Condition1_PosN', 'Condition2_PosN', 'Condition1_PosA',
        'Condition2_PosA'], axis=1)
    return data_ing


# Limpieza de datos
def limpieza(datos):
    '''This function cleans the data. It selects the variables of interest and
    transforms them to the format that will be used to train the models.

    Args:
        datos (pd.DataFrame): The data.

    Returns:
        pd.DataFrame: The cleaned data.'''
    # Eliminar columnas que no se utilizarán
    data_cln = datos[[
        'LotFrontage', 'LotArea', 'Street', 'Utilities',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
        'BsmtCond',  'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath',
        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
        'KitchenQual', 'GarageCars', 'PavedDrive', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'MiscFeature', 'SaleCondition']]

    # Conversión de unidades
    # Equivalencias
    ft_square = 0.092903  # metros cuadrados
    ft_lineal = 0.3048  # metros
    # Conversión
    data_cln['LotFrontage'] = datos['LotFrontage']*ft_lineal
    data_cln['LotArea'] = datos['LotArea']*ft_square
    data_cln['TotalBsmtSF'] = datos['TotalBsmtSF']*ft_square
    data_cln['GrLivArea'] = datos['GrLivArea']*ft_square
    data_cln['WoodDeckSF'] = datos['WoodDeckSF']*ft_square
    data_cln['OpenPorchSF'] = datos['OpenPorchSF']*ft_square
    data_cln['EnclosedPorch'] = datos['EnclosedPorch']*ft_square
    data_cln['3SsnPorch'] = datos['3SsnPorch']*ft_square
    data_cln['ScreenPorch'] = datos['ScreenPorch']*ft_square

    # Imputación de datos faltantes
    # Rellenamos colocando la opción 'NA'
    data_cln['BsmtCond'] = data_cln['BsmtCond'].fillna('NA')
    data_cln['MiscFeature'] = data_cln['MiscFeature'].fillna('NA')
    # Rellenamos con el promedio
    for i in data_cln.columns:
        if data_cln[i].dtype != 'object':
            data_cln[i].fillna(data_cln[i].mean(), inplace=True)
        else:
            data_cln[i].fillna(data_cln[i].mode()[0], inplace=True)
    return data_cln
