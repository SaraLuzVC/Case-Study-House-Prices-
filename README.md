# Tarea de la Materia de Arquitectura de Producto de Datos

Esté repositorio se creo para la materia de Arquitectura de Producto de Datos, con el porpósito de **desarrollar buenas prácticas** para construir productos de datos.

## Contexto

* Supon que estamos trabajando en una start up de bienes raices y necesitamos
construir un producto de datos que ayude a soportar una aplicación para 
que nuestros clientes (compradores/vendedores) puedan consultar una estimación
del valor de una propiedad de bienes raíces.

* Aún el CEO no tiene claro como debe de diseñarse esta aplicación. Nostros
como data scientists debemos proponer una Prueba de Concepto, que permita
experimentar rápido, dar un look an feel de la experiencia y nos permita
fallar rápido para probar una siguiente iteración.


## Objetivo:

* Prototipa un modelo en Python que permita estimar el precio de una casa
dadas algunas características que el usuario deberá proporcionar a través de
un front al momento de la inferencia.

## Datos:

* En vista de que el CEO no tiene mucha claridad, podemos construir un dataset
  con dato sintéticos o tomar alguno otro como referencia, para poder 
  desarrollar nuestra idea.

* Para lo cual usaremos el [conjunto de precios de compra-venta de casas de la
  ciudad Ames, Iowa en Estados Unidos](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Suposiciones

Debemos elegir algunas características de las casas, para que con base en éstas el usuario pueda estimar el precio. 

Como tenemos muy poca información de cómo se va a usar el producto vamos a hacer algunas suposiciones:

* La aplicación va a ser usada por personas NO especializadas en la valuación de inmuebles, por lo que habrá muchas variables que no conocerá, y por tanto no se podrán usar como input, o que se tendrán que ajustar a un menor nivel de especialización.

* Suponemos que va a ser usada por personas que quieren comprar o vender una casa o departamento y quieren darse una idea de cuanto pueden recibir.

* El producto se va a usar para el mercado de la Ciudad de México, esto implica que habrá variables que no se considerarán por el clima y cultura. Los datos que se están usando obviamente van a dar predicciones erroneas para la Ciudad de México, dado que en inmuebles la ubicación es un gran factor (debido a la demanda). Así que este sólo sería un ejercicio para jugar un poco con cómo debe ser o por dónde va la idea.

* Posteriormente se debe definir el rango de error permitido para que la predicción sea exitosa.

## Pasos

- `S01_prep.py`: Carga de la base de datos y limpieza
- `S02_built_features`: Se crean las variables  que más comúnmente se utilizan en la valuación de inmuebles en México.
- `S03_train.py`: Se entrena dos modelos vecinos más cercanos y random forest.
- `S04_inference`: Se hacen las predicciones con el modelo de Random Forest.

## Estructura del Repositorio

├── data: Carpeta con los datos de entrada (raw), despues de la limpieza, con ingeniería de variables, y finalmente las predicciones
├── models: Modelos entrenados (filetype: .sav)
├── notebooks: EDA y análisis inicial
├── references: Concurso y notebooks utilizados
├── reports: Si existe un reporte
├── requirements: Requisitos en conda
├── src: Funciones utilizadas
│   └── __pycache__
└── tests: Si llega a haber tests

## Modelo de datos
 Las variables finales que usa el modelo para predecir son:
 * 'LotFrontage': metros lineales de fachada o frente.
 * 'LotArea': metros cuadrados que mide el terreno.
 * 'Street': Terracería o pavimento.
 * 'Utilities': puede ser, todos los servicios (Electricidad, Gas, Agua y Drenaje), solo electricidad, gas y agua, solo electricidad y gas, o solo electricidad. 
 * 'YearBuilt': Año en que se construyó.
 * 'BedroomAbvGr': Numero de cuartos arriba del nivel del sótano.
 * 'KitchenQual': Calidad de la cocina, 5 niveles.
 * 'GarageCars': número de autos que caben en el garage.
 * 'PavedDrive': Entrada de autos pavimentada. Tres opciones.
 * 'OverallQual': calidad de materiales y acabados, variable ordinal con 3 niveles.
 * 'OverallCond': Condición general, variable ordinal con 3 niveles.
 * 'sotano': tiene o no sótano.
 * 'banos_completos': número de baños completos (wc, lavabo y regadera).
 * 'banos_medios': número de medios baños (wc, lavabo)
 * 'm2construidos': metros cuadrados construídos de la propiedad.
 * 'm2terraza': m2 si tiene terraza.
 * En onehot encoding:
- Ubicación, como si fuera colonia: 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker'
- Proximidad a vías principales y al ferrocarril: 'Condition1_RRNe', 'Condition_Norm', 'Condition_Feedr', 'Condition_Artery', 'Condition_PosN', 'Condition_PosA','Condition_RRAn', 'Condition_RRAe', 'Condition_RRNn'
- Tipo de prototipo, duplex, casa, etc.: 'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 
- Numero de pisos terminados:'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl'
- Miselaneo o extra: 'MiscFeature_Gar2', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC'
- Condiciones de la venta: 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial'


## Tecnología usada

- `Python`: 

## Escenario de Producción

- Los datos crudos se colocan en la carpeta de `data` con el nombre de `test.csv` y `train.csv`

## Testing
❯ pytest
==================================================================================== test session starts ====================================================================================
platform linux -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: /home/saraluz/Documents/2Sem/ArqProd/Tarea3_ArqProd
plugins: anyio-4.2.0
collected 6 items                                                                                                                                                                           

tests/test_utils.py ......                                                                                                                                                            [100%]

===================================================================================== 6 passed in 0.25s =====================================================================================
❯ 