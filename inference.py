'''This script is used to load the model
and make predictions on the test data.'''
# Cargar Bibliotecas
import pandas as pd
import joblib

# Cargo los datos
test_data_ing = pd.read_csv('./data/test_ing.csv')
train_data_ing = pd.read_csv('./data/train_ing.csv')

# Cargo el modelo
loaded_model = joblib.load('./models/rf.sav', 'rb')
predicciones = pd.DataFrame(loaded_model.predict(test_data_ing))

# Guardo predicciones
predicciones.to_csv('./data/predictions.csv')
