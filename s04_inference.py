'''This script is used to load the model
and make predictions on the test data.'''
# Cargar Bibliotecas
import argparse
import pandas as pd
import joblib
from src.utils import get_logger, no_file_error, save_file_error, num_obs

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
logger.debug("test_infile: %s", args.test_infile)
logger.debug("test_model_rf: %s", args.test_model_rf)
logger.debug("test_outfile: %s", args.test_outfile)

# Cargo los datos
logger.info("Cargando datos: %s", args.test_infile)
test_data_ing = no_file_error(args.test_infile, logger)

# Numero de observaciones mayores a 0
num_obs(test_data_ing, args.test_infile, logger)

# Cargo el modelo
# Load the Model
try:
    loaded_model = joblib.load(args.test_model_rf)
except FileNotFoundError:
    logger.error("No se pudo cargar el modelo %s", args.test_model_rf)

try:
    predictions = pd.DataFrame(loaded_model.predict(test_data_ing))
except FileNotFoundError:
    logger.error("No se pudo hacer predicciones")

# Numero de observaciones mayores a 0
num_obs(predictions, args.test_outfile, logger)

# Guardo predicciones
logger.info("Guardando datos: %s", args.test_outfile)
save_file_error(args.test_outfile, predictions, logger)
