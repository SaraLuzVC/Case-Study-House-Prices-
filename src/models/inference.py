# Cargar Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import pickle

# Cargo los datos
test_data_ing = pd.read_csv('../../data/test_ing.csv')

#Cargo el modelo
loaded_model = pickle.load(open('../../models/knn.sav', 'rb'))
# predicciones = loaded_model.predict(test_data_ing)

# Guardo predicciones
# predicciones.to_csv('../../data/predicciones.csv')