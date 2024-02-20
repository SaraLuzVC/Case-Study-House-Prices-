'''This script is used to load the model
and make predictions on the test data.'''
# Cargar Bibliotecas
import pandas as pd
import joblib
import argparse
from src.utils import get_best_score, get_logger, no_file_error, save_file_error, num_obs

# Configurar logging
logger = get_logger('inference')
logger.info('Inference starting ...')

# Cargar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/test_ing.csv')
parser.add_argument('--test_model_rf', 
                    default='./models/rf.sav')
parser.add_argument('test_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/predictions.csv')
args = parser.parse_args()
logger.debug(f"test_infile: {args.test_infile}")
logger.debug(f"test_model_rf: {args.test_model_rf}")
logger.debug(f"test_outfile: {args.test_outfile}")

# Cargo los datos
logger.info(f"Cargando datos: {args.test_infile}")
test_data_ing = no_file_error(args.test_infile, logger)

# Numero de observaciones mayores a 0
num_obs(test_data_ing, args.test_infile, logger)

# Cargo el modelo
# Load the Model
try:
    loaded_model = joblib.load(args.test_model_rf)
except Exception as e:
    logger.error(f"No se pudo cargar el modelo {args.test_model_rf}")

try:
    predictions = pd.DataFrame(loaded_model.predict(test_data_ing))
except Exception as e:
    logger.error("No se pudo hacer predicciones")    

# Numero de observaciones mayores a 0
num_obs(predictions, args.test_outfile, logger)

# Guardo predicciones
logger.info(f"Guardando datos: {args.test_outfile}")
save_file_error(args.test_outfile, predictions, logger)
