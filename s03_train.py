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
from src.utils import get_best_score

parser = argparse.ArgumentParser()
parser.add_argument('train_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/train_ing.csv')
# parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
#                     default='./data/test_cln.csv')
parser.add_argument('--test_outfile_rf', #type=argparse.FileType('w'), 
                    default='./models/rf.sav')
parser.add_argument('--test_outfile_knn', #type=argparse.FileType('w'), 
                    default='./models/knn.sav')
args = parser.parse_args()
print(args.train_infile, args.test_outfile_rf, args.test_outfile_knn)




# Cargar Datos
train_data_ing = pd.read_csv(args.train_infile)

# Abrir yaml
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
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=t_size, random_state=semilla)
X_train = X_train.drop(['SalePrice'], axis=1)
X_test = X_test.drop(['SalePrice'], axis=1)


# random forest
# configuraciones
print('Random Forest')
m_sample = config['modeling']['random_forest']['model']['min_samples_split']
print(f"min_nodo: {m_sample}")
semilla_rf = config['modeling']['random_forest']['model']['random_state']
print(f"semilla modelo: {semilla_rf}")
num_est = config['modeling']['random_forest']['model']['n_estimators']
print(f"número de árboles: {num_est}")
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
print('Vecinos más cercanos')
algoritmo = config['modeling']['knn']['model']['algorithm']
print(f"algoritmo: {algoritmo}")
num_vec = config['modeling']['knn']['model']['n_neighbors']
print(f"número de vecinos: {num_vec}")
pesos = config['modeling']['knn']['model']['weights']
print(f"pesos: {pesos}")
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

# print('Random Forest')
# print(grid_rf
#       .best_estimator_)
# print(grid_rf
#       .best_params_)
# print('KNN')
# print(grid_knn.best_params_)
# print(grid_knn.best_estimator_)

# exportar modelos
joblib.dump(grid_knn, args.test_outfile_knn)
joblib.dump(grid_rf, args.test_outfile_rf)
