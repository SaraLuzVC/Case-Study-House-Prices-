'''This script is used to load the model
and make predictions on the test data.'''
# Cargar Bibliotecas
import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/test_ing.csv')
# parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
#                     default='./data/test_cln.csv')
parser.add_argument('--test_model_rf', #type=argparse.FileType('w'), 
                    default='./models/rf.sav')
parser.add_argument('test_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/predictions.csv')
args = parser.parse_args()
#print(args.train_infile, args.test_outfile_rf, args.test_outfile_knn)





# Cargo los datos
test_data_ing = pd.read_csv(args.test_infile)

# Cargo el modelo
# Load the Model
loaded_model = joblib.load(args.test_model_rf)
predictions = pd.DataFrame(loaded_model.predict(test_data_ing))

# Guardo predicciones
predictions.to_csv(args.test_outfile, index=False)
