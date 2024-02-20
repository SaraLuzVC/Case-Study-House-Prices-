''' This code is used to clean the data and save it in a new file. '''

# Cargar Bibliotecas
import warnings
import argparse
from src.utils import limpieza, get_logger, no_file_error
from src.utils import save_file_error, num_obs, num_vars
warnings.filterwarnings("ignore")

# Configurar logging
logger = get_logger('prep')
logger.info('Prep starting ...')

# Cargar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('train_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/train.csv')
parser.add_argument('test_infile', nargs='?', type=argparse.FileType('r'),
                    default='./data/test.csv')
parser.add_argument('train_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/train_cln.csv')
parser.add_argument('test_outfile', nargs='?', type=argparse.FileType('w'),
                    default='./data/test_cln.csv')
args = parser.parse_args()
logger.debug("train_infile: %s", args.train_infile)
logger.debug("test_infile: %s", args.test_infile)
logger.debug("train_outfile: %s", args.train_outfile)
logger.debug("test_outfile: %s", args.test_outfile)

# Cargar datos
logger.info("Cargando datos: %s", args.train_infile)
train_data = no_file_error(args.train_infile, logger)
logger.info("Cargando datos: %s", args.test_infile)
test_data = no_file_error(args.test_infile, logger)

# Numero de observaciones mayores a 0
num_obs(train_data, args.train_infile, logger)
num_obs(test_data, args.test_infile, logger)

# Numero de variables
num_vars(train_data, test_data, logger)

# Limpieza de datos
logger.info("Limpieza de datos")
train_data_cln = limpieza(train_data)
test_data_cln = limpieza(test_data)

# Agregar SalePrice
TIPO_DE_CAMBIO = 17.16  # 30-ene-2024
logger.info("El tipo de cambio es %s", TIPO_DE_CAMBIO)
train_data_cln['SalePrice'] = train_data['SalePrice']
train_data_cln['SalePrice'] = round(train_data['SalePrice']*TIPO_DE_CAMBIO, 0)

# Guardar datos limpios
logger.info("Guardando datos: %s", args.train_outfile)
save_file_error(args.train_outfile, train_data_cln, logger)
logger.info("Guardando datos: %s", args.test_outfile)
save_file_error(args.test_outfile, test_data_cln, logger)
