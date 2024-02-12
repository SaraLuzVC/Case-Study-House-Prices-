# Cargar Bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import pickle

# Cargar Datos
train_data_ing = pd.read_csv('../../data/train_ing.csv')

# Asignar variables
X = train_data_ing
y = train_data_ing['SalePrice']

# Dividir datos en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.drop(['SalePrice'], axis=1)
X_test = X_test.drop(['SalePrice'], axis=1)

# Función para obtener el mejor score
def get_best_score(grid):
    best_score = np.sqrt(-grid.best_score_)
    print(best_score)    
    print(grid.best_params_)
    print(grid.best_estimator_)
    return best_score

# random forest
score_calc = 'neg_mean_squared_error'
param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, refit=True, verbose=1, scoring = score_calc)
grid_rf.fit(X_train, y_train)
sc_rf = get_best_score(grid_rf)


# vecinos más cercanos
param_grid = {'n_neighbors' : [3,4,5,6,7,10,15] ,    
              'weights' : ['uniform','distance'] ,
              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}
grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, refit=True, verbose=1, scoring = score_calc)
grid_knn.fit(X_train, y_train)
sc_knn = get_best_score(grid_knn)

# crear pickle
pickle.dump(grid_knn, open('../../models/knn.sav', 'wb'))
pickle.dump(grid_rf, open('../../models/grid_rf.sav', 'wb'))

