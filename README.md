Utilitario de Regresión Lineal en AWS
=====================================

Este proyecto implementa un utilitario para realizar regresión lineal utilizando servicios de AWS, incluyendo Amazon Athena, Apache Spark y Amazon SageMaker.

Contenido
---------

1.  [Preparación de Datos con Apache Spark](#preparación-de-datos-con-apache-spark)
    
2.  [Entrenamiento del Modelo con Amazon SageMaker](#entrenamiento-del-modelo-con-amazon-sagemaker)
    
3.  [Configuración y Uso](#configuración-y-uso)
    

Preparación de Datos con Apache Spark
-------------------------------------

El archivo de preparación de datos utiliza PySpark para procesar y preparar los datos para el entrenamiento del modelo.

### Pasos Principales:

1.  **Configuración del Entorno**:
    
    *   Se crea un Notebook en PySpark sobre Athena.
        
    *   Se importan las librerías necesarias, incluyendo tipos de datos de PySpark y pandas compatible con Big Data.
  
  
       ```yaml
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
        import pyspark.pandas as pd
        pd.set_option("display.max_rows", 20)
        import boto3
       ```        
3.  **Configuración del Archivo de Entrada**:
    
    *   Se define el bucket de S3 y la ruta del archivo.
        
    *   Se establece el esquema de metadatos para los campos del dataset.
        
    *   Se definen las variables categóricas y el campo label.
        
4.  **Lectura y Procesamiento de Datos**:
    
    *   Se lee el archivo CSV desde S3.
        
    *   Se convierte el DataFrame Spark a Pandas para procesamiento.
        
    *   Se crean columnas dummy para variables categóricas.
        
5.  **Preparación Final de Datos**:
    
    *   Se ordena el DataFrame y se divide en conjuntos de entrenamiento y validación.
        
    *   Se almacenan los DataFrames resultantes en S3.
        
6.  **Limpieza**:
    
    *   Se eliminan archivos temporales "\_SUCCESS" de S3.
        

Entrenamiento del Modelo con Amazon SageMaker
---------------------------------------------

El utilitario de SageMaker se encarga de configurar y entrenar el modelo de regresión lineal.

### Pasos Principales:

1.  **Configuración Inicial**:
    
    *   Se inicia una sesión de SageMaker y se obtiene el rol de ejecución.
        
    *   Se configura la lectura de datos de entrenamiento y validación desde S3.
        
2.  **Configuración del Modelo**:
    
    *   Se define el nombre del job de entrenamiento y el algoritmo (linear-learner).
        
    *   Se especifican parámetros como el tipo de predicción, número de servidores, y tipo de instancia.
        
3.  **Entrenamiento del Modelo**:
    
    *   Se configura el estimador con hiperparámetros específicos.
        
    *   Se inicia el proceso de entrenamiento con los datos de entrenamiento y validación.
        
4.  **Evaluación del Modelo**:
    
    *   Se obtienen y analizan las métricas del modelo entrenado, incluyendo MSE y R².
        

Configuración y Uso
-------------------

1.  **Requisitos Previos**:
    
    *   Cuenta de AWS con acceso a S3, Athena, y SageMaker.
        
    *   Python 3.x y las librerías necesarias instaladas.
        
2.  **Configuración**:
    
    *   Asegúrese de tener las credenciales de AWS configuradas correctamente.
        
    *   Modifique las variables de bucket y rutas de S3 según su configuración.
        
3.  **Ejecución**:
    
    *   Ejecute primero el notebook de preparación de datos en un entorno PySpark.
        
    *   Luego, ejecute el notebook de SageMaker para entrenar y evaluar el modelo.
        
4.  **Monitoreo**:
    
    *   Utilice la consola de AWS SageMaker para monitorear el progreso del entrenamiento y ver las métricas en tiempo real.
        

Notas Importantes
-----------------

*   Asegúrese de tener los permisos necesarios en IAM para acceder a los servicios de AWS utilizados.
    
*   Los costos asociados con el uso de estos servicios de AWS pueden aplicar.
    
*   Revise y ajuste los hiperparámetros del modelo según sea necesario para su caso de uso específico.
