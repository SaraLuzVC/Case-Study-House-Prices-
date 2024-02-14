'''This script is used to load the model
and make predictions on the test data.'''
# Cargar Bibliotecas
import pandas as pd
import joblib

# Cargo los datos
test_data_ing = pd.read_csv('./data/test_ing.csv')

# Cargo el modelo
# Load the Model
loaded_model = joblib.load('./models/rf.sav')
predictions = pd.DataFrame(loaded_model.predict(test_data_ing))

# Guardo predicciones
predictions.to_csv('./data/predictions.csv', index=False)
