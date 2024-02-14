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
 


## Tecnología usada

- `Python`: 

## Escenario de Producción

- Assuming that Yelp data can be extracted periodically, this pipeline can be executed in AWS services using a EMR cluster with Spark.

- Data should be stored in the data lake in S3, using the code `S00_stage_data.ipynb`.

- The ETL in `S01_etl.ipynb` already has the code to launch the EMR cluster, and in `config` the configuration files to setup the credentials and variables for the infrastructure will be stored.

- Based on the objective of this analysis, the pipeline could be scheduled to run with Airflow every start of a new month.
