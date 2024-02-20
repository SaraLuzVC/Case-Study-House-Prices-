'''This code is used to train the models and save them in a file.
We use two models: a random forest and a k-nearest neighbors model. We use
grid search to find the best hyperparameters for each model. We then save the
best models in a file. We also use the root mean squared error as the scoring
metric for the grid search. We use the training data to train the models.
'''
# Cargar Bibliotecas
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import joblib
import yaml
from src.utils import get_best_score, get_logger, no_file_error, save_file_error, num_obs

# Configurar logging
logger = get_logger('train')
logger.info('Training Models ...')

# Cargar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('train_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/train_ing.csv')
parser.add_argument('--test_outfile_rf',
                    default='./models/rf.sav')
parser.add_argument('--test_outfile_knn',
                    default='./models/knn.sav')
args = parser.parse_args()
logger.debug(f"train_infile: {args.train_infile}")
logger.debug(f"test_outfile_rf: {args.test_outfile_rf}")
logger.debug(f"test_outfile_knn: {args.test_outfile_knn}")

# Cargar Datos
logger.info(f"Cargando datos: {args.train_infile}")
train_data_ing = no_file_error(args.train_infile, logger)

# Numero de observaciones mayores a 0
num_obs(train_data_ing, args.train_infile, logger)

# Abrir yaml
logger.info("Cargando configuraciones")
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Leer configuraciones
print("\nProject name:")
print(config['main']['project_name'])
semilla = config['modeling']['random_seed']
print(f"semilla: {semilla}")
t_size = config['modeling']['test_size']
print(f"tamaño de test: {t_size}")

# Asignar variables
X = train_data_ing
y = train_data_ing['SalePrice']

# Dividir datos en training y test
logger.info("Dividiendo datos en training y test")
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=t_size, random_state=semilla)
X_train = X_train.drop(['SalePrice'], axis=1)
X_test = X_test.drop(['SalePrice'], axis=1)


# random forest
# configuraciones
logger.info('Random Forest')
m_sample = config['modeling']['random_forest']['model']['min_samples_split']
logger.debug(f"min_nodo: {m_sample}")
semilla_rf = config['modeling']['random_forest']['model']['random_state']
logger.debug(f"semilla modelo: {semilla_rf}")
num_est = config['modeling']['random_forest']['model']['n_estimators']
logger.debug(f"número de árboles: {num_est}")
SCORE_CALC = 'neg_mean_squared_error'
param_grid = {'min_samples_split': [m_sample],
               'n_estimators': [num_est], 'random_state': [semilla_rf]}
# modelo
grid_rf = GridSearchCV(RandomForestRegressor(),
                       param_grid, cv=5,
                       refit=True, verbose=1,
                       scoring=SCORE_CALC)
grid_rf.fit(X_train, y_train)
sc_rf = get_best_score(grid_rf)

# vecinos más cercanos
# configuraciones
logger.info('Vecinos más cercanos')
algoritmo = config['modeling']['knn']['model']['algorithm']
logger.debug(f"algoritmo: {algoritmo}")
num_vec = config['modeling']['knn']['model']['n_neighbors']
logger.debug(f"número de vecinos: {num_vec}")
pesos = config['modeling']['knn']['model']['weights']
logger.debug(f"pesos: {pesos}")
param_grid = {'n_neighbors': [num_vec],
              'weights': [pesos],
              'algorithm': [algoritmo]}
# modelo
grid_knn = GridSearchCV(KNeighborsRegressor(),
                        param_grid, cv=5,
                        refit=True, verbose=1,
                        scoring=SCORE_CALC)
grid_knn.fit(X_train, y_train)
sc_knn = get_best_score(grid_knn)

logger.info('Random Forest')
logger.info(grid_rf.best_params_)
logger.info(grid_rf.best_estimator_)
logger.info('KNN')
logger.info(grid_knn.best_params_)
logger.info(grid_knn.best_estimator_)

# exportar modelos
try:
    joblib.dump(grid_rf, args.test_outfile_rf)
except Exception as e:
    logger.error(f"No se pudo guardar el archivo {args.test_outfile_rf}")
try:
    joblib.dump(grid_rf, args.test_outfile_rf)
except Exception as e:
    logger.error(f"No se pudo guardar el archivo {args.test_outfile_rf}")
